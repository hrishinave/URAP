from __future__ import annotations

import csv
import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable

from .classifier import classify_text
from .config import MiningConfig
from .evidence import find_fix_evidence
from .github import DEFAULT_PER_PAGE, GitHubClient
from .queries import SearchQuerySpec, build_query_specs, build_search_query


MAX_RESULTS_PER_QUERY = 1000
logger = logging.getLogger(__name__)

CANDIDATE_FIELDS = [
    "repo",
    "issue_number",
    "issue_url",
    "title",
    "state",
    "created_at",
    "closed_at",
    "updated_at",
    "matched_queries",
    "matched_terms",
    "suspected_trigger",
    "prior_work_match",
    "novelty_status",
    "evidence_type",
    "linked_fix_url",
    "evidence_detail",
    "labels",
    "notes",
    "is_valid",
    "trigger_category",
    "fp_order_mechanism",
    "fix_summary",
    "invalid_reason",
]


def collect_candidates(
    config: MiningConfig,
    client: GitHubClient,
    repos: list[str] | None = None,
    max_queries: int | None = None,
    include_without_fix: bool = False,
    fetch_comments: bool = True,
    per_page: int = DEFAULT_PER_PAGE,
    on_row: Callable[[dict[str, str]], None] | None = None,
    search_cache_path: str | Path | None = None,
    resume_search: bool = False,
) -> list[dict[str, str]]:
    specs = build_query_specs(config, repos=repos)
    if max_queries is not None:
        specs = specs[:max_queries]

    issue_matches: dict[str, dict[str, Any]] = {}
    completed_queries: set[str] = set()
    if resume_search and search_cache_path is not None and Path(search_cache_path).exists():
        issue_matches, completed_queries = read_search_cache(search_cache_path)
        logger.info(
            "Loaded search checkpoint with %d completed queries and %d unique candidate issues from %s",
            len(completed_queries),
            len(issue_matches),
            search_cache_path,
        )

    for index, spec in enumerate(specs, start=1):
        query = build_search_query(spec, config.start_date, config.end_date)
        if query in completed_queries:
            logger.info("Skipping cached query %d/%d for %s: %s", index, len(specs), spec.repo, ", ".join(spec.terms))
            continue

        logger.info("Running query %d/%d for %s: %s", index, len(specs), spec.repo, ", ".join(spec.terms))
        items = search_with_bisection(client, spec, config.start_date, config.end_date, per_page=per_page)
        for item in items:
            repo = repo_from_item(item) or spec.repo
            number = int(item["number"])
            key = f"{repo}#{number}"
            match = issue_matches.setdefault(
                key,
                {
                    "repo": repo,
                    "number": number,
                    "item": item,
                    "matched_terms": set(),
                    "matched_queries": set(),
                },
            )
            match["matched_terms"].update(spec.terms)
            match["matched_queries"].add(query)

        completed_queries.add(query)
        if search_cache_path is not None:
            write_search_cache(issue_matches, completed_queries, search_cache_path)

    logger.info("Search phase found %d unique candidate issues; fetching details and fix evidence", len(issue_matches))
    rows = []
    for index, match in enumerate(issue_matches.values(), start=1):
        logger.info("Enriching issue %d/%d: %s#%s", index, len(issue_matches), match["repo"], match["number"])
        row = build_candidate_row(config, client, match, include_without_fix=include_without_fix, fetch_comments=fetch_comments)
        if row is not None:
            rows.append(row)
            if on_row is not None:
                on_row(row)
    rows.sort(key=lambda row: (row["repo"].lower(), int(row["issue_number"])))
    return rows


def search_with_bisection(
    client: GitHubClient,
    spec: SearchQuerySpec,
    start_date: date,
    end_date: date,
    per_page: int = DEFAULT_PER_PAGE,
    max_results_per_query: int = MAX_RESULTS_PER_QUERY,
) -> list[dict[str, Any]]:
    stack = [(start_date, end_date)]
    items: list[dict[str, Any]] = []
    while stack:
        start, end = stack.pop()
        query = build_search_query(spec, start, end)
        page_items, total_count, headers = client.search_issues_page(query, per_page=per_page)
        if total_count > max_results_per_query and start < end:
            mid = start + (end - start) // 2
            stack.append((mid + timedelta(days=1), end))
            stack.append((start, mid))
            continue
        items.extend(page_items)
        items.extend(client.paginate_from_headers(headers))
    return items


def build_candidate_row(
    config: MiningConfig,
    client: GitHubClient,
    match: dict[str, Any],
    include_without_fix: bool = False,
    fetch_comments: bool = True,
) -> dict[str, str] | None:
    repo = match["repo"]
    number = match["number"]
    item = match["item"]

    issue = client.get_issue(repo, number)
    if issue.get("pull_request"):
        return None
    comments = client.get_issue_comments(repo, number) if fetch_comments else []

    full_text = "\n".join(
        [_text(issue.get("title")), _text(issue.get("body"))]
        + [_text(comment.get("body")) for comment in comments]
    )
    classification = classify_text(full_text, config)
    evidence = find_fix_evidence(repo, issue, comments, client=client)
    if not include_without_fix and not evidence.has_evidence:
        return None

    matched_terms = set(match["matched_terms"])
    matched_terms.update(classification.matched_terms)
    labels = [_text(label.get("name")) for label in issue.get("labels", []) if label.get("name")]

    trigger_category = ";".join(classification.suspected_trigger)
    return {
        "repo": repo,
        "issue_number": str(number),
        "issue_url": _text(issue.get("html_url") or item.get("html_url")),
        "title": _text(issue.get("title") or item.get("title")),
        "state": _text(issue.get("state", item.get("state", ""))),
        "created_at": _text(issue.get("created_at", item.get("created_at", ""))),
        "closed_at": _text(issue.get("closed_at")),
        "updated_at": _text(issue.get("updated_at", item.get("updated_at", ""))),
        "matched_queries": "\n".join(sorted(match["matched_queries"])),
        "matched_terms": ";".join(sorted(matched_terms, key=str.lower)),
        "suspected_trigger": trigger_category,
        "prior_work_match": ";".join(classification.prior_work_match),
        "novelty_status": classification.novelty_status,
        "evidence_type": evidence.evidence_type,
        "linked_fix_url": evidence.linked_fix_url,
        "evidence_detail": evidence.evidence_detail,
        "labels": ";".join(labels),
        "notes": "",
        "is_valid": "",
        "trigger_category": trigger_category,
        "fp_order_mechanism": "",
        "fix_summary": "",
        "invalid_reason": "",
    }


def write_candidates_csv(rows: list[dict[str, str]], path: str | Path) -> None:
    write_csv(rows, path, CANDIDATE_FIELDS)


def append_candidate_csv(row: dict[str, str], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CANDIDATE_FIELDS, extrasaction="ignore")
        writer.writerow(row)


def write_search_cache(issue_matches: dict[str, dict[str, Any]], completed_queries: set[str], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "completed_queries": sorted(completed_queries),
        "issues": [
            {
                "repo": match["repo"],
                "number": match["number"],
                "item": match["item"],
                "matched_terms": sorted(match["matched_terms"]),
                "matched_queries": sorted(match["matched_queries"]),
            }
            for match in issue_matches.values()
        ],
    }
    temp_path = output.with_suffix(output.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    temp_path.replace(output)


def read_search_cache(path: str | Path) -> tuple[dict[str, dict[str, Any]], set[str]]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)

    issue_matches = {}
    for match in payload.get("issues", []):
        key = f"{match['repo']}#{match['number']}"
        issue_matches[key] = {
            "repo": match["repo"],
            "number": int(match["number"]),
            "item": match["item"],
            "matched_terms": set(match.get("matched_terms", [])),
            "matched_queries": set(match.get("matched_queries", [])),
        }
    return issue_matches, set(payload.get("completed_queries", []))


def export_validated_cases(review_csv: str | Path, output_csv: str | Path) -> list[dict[str, str]]:
    with Path(review_csv).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [
            row
            for row in reader
            if _truthy(row.get("is_valid", "")) and row.get("evidence_type", "").strip()
        ]
    write_csv(rows, output_csv, rows[0].keys() if rows else CANDIDATE_FIELDS)
    return rows


def write_csv(rows: list[dict[str, str]], path: str | Path, fieldnames: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    field_list = list(fieldnames)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_list, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def repo_from_item(item: dict[str, Any]) -> str:
    repo_url = item.get("repository_url", "")
    if repo_url:
        return "/".join(repo_url.rstrip("/").split("/")[-2:])
    html_url = item.get("html_url", "")
    parts = html_url.split("/")
    if len(parts) >= 5 and parts[2] == "github.com":
        return f"{parts[3]}/{parts[4]}"
    return ""


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "valid"}
