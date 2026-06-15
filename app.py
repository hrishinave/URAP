import logging
import os
import re
import time
from datetime import date, datetime, timedelta

import requests as http_requests
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GITHUB_API_SEARCH_URL = "https://api.github.com/search/issues"
MAX_RESULTS_PER_QUERY = 1000
DEFAULT_PER_PAGE = 100

# ---------------------------------------------------------------------------
# Rate-limit handling
# ---------------------------------------------------------------------------

_rate_remaining = None
_rate_reset = None
_backoff_count = 0


def _update_rate_limit(headers):
    global _rate_remaining, _rate_reset
    if "X-RateLimit-Remaining" in headers:
        _rate_remaining = int(headers["X-RateLimit-Remaining"])
    if "X-RateLimit-Reset" in headers:
        _rate_reset = float(headers["X-RateLimit-Reset"])


def _wait_if_needed():
    if _rate_remaining is not None and _rate_remaining == 0 and _rate_reset is not None:
        sleep_for = max(_rate_reset - time.time() + 1, 0)
        if sleep_for > 0:
            logger.info("Rate limit exhausted, sleeping %.1fs", sleep_for)
            time.sleep(sleep_for)


def _handle_rate_limit_response(resp):
    global _backoff_count

    # 1) Retry-After header
    retry_after = resp.headers.get("Retry-After")
    if retry_after:
        sleep_for = int(retry_after) + 1
        logger.info("Retry-After header: sleeping %ds", sleep_for)
        time.sleep(sleep_for)
        return

    # 2) X-RateLimit-Remaining == 0
    remaining = resp.headers.get("X-RateLimit-Remaining")
    reset_at = resp.headers.get("X-RateLimit-Reset")
    if remaining and int(remaining) == 0 and reset_at:
        sleep_for = max(float(reset_at) - time.time() + 1, 0)
        logger.info("Primary rate limit hit, sleeping %.1fs", sleep_for)
        time.sleep(sleep_for)
        return

    # 3) Secondary / abuse limit – exponential backoff
    sleep_for = min(2 ** _backoff_count * 10, 300)
    _backoff_count += 1
    logger.warning("Secondary rate limit, backing off %ds (attempt %d)", sleep_for, _backoff_count)
    time.sleep(sleep_for)


def _reset_backoff():
    global _backoff_count
    _backoff_count = 0


# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------

def _github_headers():
    token = os.environ.get("GITHUB_TOKEN", "")
    h = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def make_request(url, params=None, max_retries=5):
    """GET request to GitHub with rate-limit handling and retries."""
    headers = _github_headers()
    for attempt in range(max_retries):
        _wait_if_needed()
        try:
            resp = http_requests.get(url, params=params, headers=headers, timeout=30)
        except http_requests.RequestException as exc:
            logger.warning("Network error (attempt %d): %s", attempt + 1, exc)
            time.sleep(5)
            continue

        if resp.status_code == 200:
            _update_rate_limit(resp.headers)
            _reset_backoff()
            return resp

        if resp.status_code in (403, 429):
            _update_rate_limit(resp.headers)
            _handle_rate_limit_response(resp)
            continue

        # Other errors – log the body so we can debug, then raise
        try:
            body = resp.json()
        except ValueError:
            body = resp.text
        logger.error("HTTP %d from %s: %s", resp.status_code, url, body)
        resp.raise_for_status()

    raise RuntimeError(f"Exhausted {max_retries} retries for {url}")


def parse_link_header(header):
    """Parse Link header into {rel: url} dict."""
    links = {}
    if not header:
        return links
    for part in header.split(","):
        match = re.match(r'\s*<([^>]+)>;\s*rel="(\w+)"', part.strip())
        if match:
            links[match.group(2)] = match.group(1)
    return links


# ---------------------------------------------------------------------------
# Search logic
# ---------------------------------------------------------------------------

def paginate_query(query, per_page=DEFAULT_PER_PAGE):
    """Run a search query and paginate through all result pages (up to 1000)."""
    params = {"q": query, "per_page": per_page, "sort": "created", "order": "asc"}
    resp = make_request(GITHUB_API_SEARCH_URL, params=params)
    data = resp.json()
    total_count = data.get("total_count", 0)
    items = list(data.get("items", []))

    # Follow next pages
    next_url = parse_link_header(resp.headers.get("Link", "")).get("next")
    while next_url:
        resp = make_request(next_url)
        page_data = resp.json()
        items.extend(page_data.get("items", []))
        next_url = parse_link_header(resp.headers.get("Link", "")).get("next")

    return items, total_count


def _search_repo(keywords, repo, start_date, end_date, state=None, per_page=DEFAULT_PER_PAGE):
    """Search issues in a specific repo with date-range bisection."""
    all_items = []
    stack = [(start_date, end_date)]
    keyword_part = " ".join(f'"{kw}"' for kw in keywords)
    state_part = f" is:{state}" if state else ""

    while stack:
        s, e = stack.pop()
        query = f'{keyword_part} is:issue repo:{repo}{state_part} created:{s}..{e}'
        logger.info("Searching: %s", query)

        # Fetch first page to check total_count
        params = {"q": query, "per_page": per_page, "sort": "created", "order": "asc"}
        resp = make_request(GITHUB_API_SEARCH_URL, params=params)
        data = resp.json()
        total_count = data.get("total_count", 0)

        if total_count == 0:
            logger.info("  -> 0 results, skipping")
            continue

        if total_count <= MAX_RESULTS_PER_QUERY:
            # Safe – collect this page and paginate the rest
            items = list(data.get("items", []))
            next_url = parse_link_header(resp.headers.get("Link", "")).get("next")
            while next_url:
                resp = make_request(next_url)
                page_data = resp.json()
                items.extend(page_data.get("items", []))
                next_url = parse_link_header(resp.headers.get("Link", "")).get("next")
            all_items.extend(items)
            logger.info("  -> %d results collected", len(items))
        else:
            # Need to bisect
            if s == e:
                # Single day, can't split further – grab what we can
                items = list(data.get("items", []))
                next_url = parse_link_header(resp.headers.get("Link", "")).get("next")
                while next_url:
                    resp = make_request(next_url)
                    page_data = resp.json()
                    items.extend(page_data.get("items", []))
                    next_url = parse_link_header(resp.headers.get("Link", "")).get("next")
                all_items.extend(items)
                logger.warning("  -> Single day %s has %d results (>1000), captured %d", s, total_count, len(items))
            else:
                mid = s + (e - s) // 2
                logger.info("  -> %d results, bisecting at %s", total_count, mid)
                stack.append((mid + timedelta(days=1), e))
                stack.append((s, mid))

    return all_items


def search_github(keywords, repos, start_date, end_date, state=None, per_page=DEFAULT_PER_PAGE):
    """Search issues across the given repos."""
    items = []
    errors = []
    for repo in repos:
        logger.info("Searching repo: %s for keywords: %s state: %s", repo, keywords, state or "all")
        try:
            items += _search_repo(keywords, repo, start_date, end_date, state=state, per_page=per_page)
        except Exception as exc:
            msg = f"{repo}: {exc}"
            logger.error("Failed to search %s", msg)
            errors.append(msg)
    return items, errors


def transform_item(item, keyword):
    """Extract the fields we care about from a GitHub search result item."""
    # Repo name: parse from repository_url if available
    repo_url = item.get("repository_url", "")
    repo = "/".join(repo_url.rstrip("/").split("/")[-2:]) if repo_url else "unknown"

    return {
        "repo": repo,
        "title": item.get("title", ""),
        "url": item.get("html_url", ""),
        "created_at": item.get("created_at", ""),
        "state": item.get("state", ""),
        "keyword": keyword,
    }


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/search")
def api_search():
    keyword = request.args.get("q", "").strip()
    repos_raw = request.args.get("repos", "").strip()
    start = request.args.get("start_date", "").strip()
    end = request.args.get("end_date", "").strip()
    state = request.args.get("state", "").strip() or None

    if not keyword:
        return jsonify({"error": "Missing keyword(s)"}), 400
    if not repos_raw:
        return jsonify({"error": "Missing repos"}), 400

    # Parse repos: accept "owner/repo" from URLs or direct input, comma/newline separated
    repos = []
    for raw in re.split(r"[,\n]+", repos_raw):
        raw = raw.strip().rstrip("/")
        if not raw:
            continue
        # Handle full GitHub URLs like https://github.com/owner/repo
        match = re.search(r"github\.com/([^/]+/[^/]+)", raw)
        if match:
            repos.append(match.group(1))
        elif "/" in raw:
            repos.append(raw)
    if not repos:
        return jsonify({"error": "No valid repos provided. Use owner/repo format or GitHub URLs."}), 400

    try:
        start_date = datetime.strptime(start, "%Y-%m-%d").date() if start else date.today() - timedelta(days=30)
        end_date = datetime.strptime(end, "%Y-%m-%d").date() if end else date.today()
    except ValueError:
        return jsonify({"error": "Invalid date format, use YYYY-MM-DD"}), 400

    if start_date > end_date:
        return jsonify({"error": "start_date must be before end_date"}), 400

    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        return jsonify({"error": "GITHUB_TOKEN environment variable not set"}), 500

    # Parse keywords: comma-separated
    keywords = [k.strip() for k in keyword.split(",") if k.strip()]
    if not keywords:
        return jsonify({"error": "Missing keyword(s)"}), 400

    try:
        raw_items, search_errors = search_github(keywords, repos, start_date, end_date, state=state)
    except Exception as exc:
        logger.exception("Search failed")
        return jsonify({"error": str(exc)}), 500

    # Deduplicate by URL
    seen = set()
    results = []
    keyword_label = ", ".join(keywords)
    for item in raw_items:
        url = item.get("html_url", "")
        if url in seen:
            continue
        seen.add(url)
        results.append(transform_item(item, keyword_label))

    resp = {"total": len(results), "results": results}
    if search_errors:
        resp["warnings"] = search_errors
    return jsonify(resp)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
