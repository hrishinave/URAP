from __future__ import annotations

import logging
import os
import re
import time
from typing import Any


GITHUB_API_SEARCH_URL = "https://api.github.com/search/issues"
GITHUB_API_BASE_URL = "https://api.github.com"
DEFAULT_PER_PAGE = 100

logger = logging.getLogger(__name__)


class GitHubNotFoundError(RuntimeError):
    """Raised for GitHub 404s that callers may intentionally treat as absent."""


def parse_link_header(header: str | None) -> dict[str, str]:
    links: dict[str, str] = {}
    if not header:
        return links
    for part in header.split(","):
        match = re.match(r'\s*<([^>]+)>;\s*rel="(\w+)"', part.strip())
        if match:
            links[match.group(2)] = match.group(1)
    return links


class GitHubClient:
    def __init__(
        self,
        token: str | None = None,
        session: Any | None = None,
        sleep=time.sleep,
        timeout: int = 30,
        api_base_url: str = GITHUB_API_BASE_URL,
    ) -> None:
        self.token = token if token is not None else os.environ.get("GITHUB_TOKEN", "")
        if not self.token:
            raise ValueError("GITHUB_TOKEN environment variable not set")
        if session is None:
            import requests

            session = requests.Session()
        self.session = session
        self.sleep = sleep
        self.timeout = timeout
        self.api_base_url = api_base_url.rstrip("/")
        self._rate_remaining: int | None = None
        self._rate_reset: float | None = None
        self._backoff_count = 0

    def get_json(self, url: str, params: dict[str, Any] | None = None, max_retries: int = 5) -> tuple[Any, dict[str, str]]:
        resp = self._get(url, params=params, max_retries=max_retries)
        return resp.json(), dict(resp.headers)

    def paginate(self, url: str, params: dict[str, Any] | None = None) -> tuple[list[Any], dict[str, str]]:
        data, headers = self.get_json(url, params=params)
        items = list(data if isinstance(data, list) else data.get("items", []))
        next_url = parse_link_header(headers.get("Link")).get("next")
        last_headers = headers
        while next_url:
            page_data, last_headers = self.get_json(next_url)
            items.extend(page_data if isinstance(page_data, list) else page_data.get("items", []))
            next_url = parse_link_header(last_headers.get("Link")).get("next")
        return items, last_headers

    def search_issues(self, query: str, per_page: int = DEFAULT_PER_PAGE) -> tuple[list[dict[str, Any]], int]:
        items, total_count, headers = self.search_issues_page(query, per_page=per_page)
        items.extend(self.paginate_from_headers(headers))
        return items, total_count

    def search_issues_page(self, query: str, per_page: int = DEFAULT_PER_PAGE) -> tuple[list[dict[str, Any]], int, dict[str, str]]:
        params = {"q": query, "per_page": per_page, "sort": "created", "order": "asc"}
        data, headers = self.get_json(GITHUB_API_SEARCH_URL, params=params)
        items = list(data.get("items", []))
        total_count = int(data.get("total_count", 0))
        return items, total_count, headers

    def paginate_from_headers(self, headers: dict[str, str]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        next_url = parse_link_header(headers.get("Link")).get("next")
        while next_url:
            page_data, headers = self.get_json(next_url)
            items.extend(page_data.get("items", []))
            next_url = parse_link_header(headers.get("Link")).get("next")
        return items

    def get_issue(self, repo: str, number: int) -> dict[str, Any]:
        data, _ = self.get_json(f"{self.api_base_url}/repos/{repo}/issues/{number}")
        return data

    def get_issue_comments(self, repo: str, number: int) -> list[dict[str, Any]]:
        comments, _ = self.paginate(f"{self.api_base_url}/repos/{repo}/issues/{number}/comments", params={"per_page": 100})
        return comments

    def get_pull(self, repo: str, number: int) -> dict[str, Any]:
        data, _ = self.get_json(f"{self.api_base_url}/repos/{repo}/pulls/{number}")
        return data

    def _get(self, url: str, params: dict[str, Any] | None = None, max_retries: int = 5) -> Any:
        headers = self._headers()
        for attempt in range(max_retries):
            self._wait_if_needed()
            try:
                resp = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
            except Exception as exc:
                logger.warning("Network error on attempt %d for %s: %s", attempt + 1, url, exc)
                self.sleep(5)
                continue

            if resp.status_code == 200:
                self._update_rate_limit(resp.headers)
                self._backoff_count = 0
                return resp

            if resp.status_code in (403, 429):
                self._update_rate_limit(resp.headers)
                self._handle_rate_limit_response(resp)
                continue

            if resp.status_code == 404:
                raise GitHubNotFoundError(f"GitHub resource not found: {url}")

            logger.error("HTTP %d from %s: %s", resp.status_code, url, _response_body(resp))
            resp.raise_for_status()

        raise RuntimeError(f"Exhausted {max_retries} retries for {url}")

    def _headers(self) -> dict[str, str]:
        return {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Authorization": f"Bearer {self.token}",
        }

    def _update_rate_limit(self, headers: dict[str, Any]) -> None:
        if "X-RateLimit-Remaining" in headers:
            self._rate_remaining = int(headers["X-RateLimit-Remaining"])
        if "X-RateLimit-Reset" in headers:
            self._rate_reset = float(headers["X-RateLimit-Reset"])

    def _wait_if_needed(self) -> None:
        if self._rate_remaining == 0 and self._rate_reset is not None:
            sleep_for = max(self._rate_reset - time.time() + 1, 0)
            if sleep_for > 0:
                logger.info("GitHub rate limit exhausted, sleeping %.1fs", sleep_for)
                self.sleep(sleep_for)

    def _handle_rate_limit_response(self, resp: Any) -> None:
        retry_after = resp.headers.get("Retry-After")
        if retry_after:
            sleep_for = int(retry_after) + 1
            logger.info("GitHub Retry-After header received, sleeping %ds", sleep_for)
            self.sleep(sleep_for)
            return

        remaining = resp.headers.get("X-RateLimit-Remaining")
        reset_at = resp.headers.get("X-RateLimit-Reset")
        if remaining and int(remaining) == 0 and reset_at:
            sleep_for = max(float(reset_at) - time.time() + 1, 0)
            logger.info("GitHub primary rate limit hit, sleeping %.1fs", sleep_for)
            self.sleep(sleep_for)
            return

        sleep_for = min(2 ** self._backoff_count * 10, 300)
        self._backoff_count += 1
        logger.warning("GitHub secondary rate limit likely hit, backing off %ds", sleep_for)
        self.sleep(sleep_for)


def _response_body(resp: Any) -> Any:
    try:
        return resp.json()
    except ValueError:
        return resp.text
