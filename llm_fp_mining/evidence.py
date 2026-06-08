from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


FIXED_VERSION_RE = re.compile(r"\b(?:fixed|resolved|patched|released|available)\s+in\s+(v?\d+(?:\.\d+){1,3}[^\s,;)]+)?", re.I)
FIX_REF_RE = re.compile(
    r"\b(?:fix(?:e[sd])?|clos(?:e[sd])?|resolv(?:e[sd])?|address(?:ed|es)?|patched|implemented)\b"
    r"(?:\s+\w+){0,5}\s+#(?P<number>\d+)",
    re.I,
)
PR_URL_RE = re.compile(
    r"https://github\.com/(?P<repo>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/pull/(?P<number>\d+)",
    re.I,
)
MAINTAINER_ASSOCIATIONS = {"OWNER", "MEMBER", "COLLABORATOR"}


@dataclass(frozen=True)
class FixEvidence:
    evidence_type: str = ""
    linked_fix_url: str = ""
    evidence_detail: str = ""

    @property
    def has_evidence(self) -> bool:
        return bool(self.evidence_type)


def find_fix_evidence(
    repo: str,
    issue: dict[str, Any],
    comments: list[dict[str, Any]],
    client: Any | None = None,
) -> FixEvidence:
    texts = _issue_texts(issue, comments)
    pr_numbers = extract_pr_numbers(repo, texts)
    for number in pr_numbers:
        if client is None:
            return FixEvidence("linked_pr_mentioned", f"https://github.com/{repo}/pull/{number}", f"PR #{number} mentioned")
        try:
            pull = client.get_pull(repo, number)
        except Exception:
            continue
        if pull.get("merged_at"):
            return FixEvidence("linked_merged_pr", pull.get("html_url", ""), f"PR #{number} is merged")
        return FixEvidence("linked_pr_unmerged", pull.get("html_url", ""), f"PR #{number} is linked but not merged")

    fixed_version = find_fixed_version_reference(texts)
    if fixed_version:
        return fixed_version

    maintainer = find_maintainer_confirmation(comments)
    if maintainer:
        return maintainer

    return FixEvidence()


def extract_pr_numbers(repo: str, texts: list[str]) -> list[int]:
    numbers: list[int] = []
    seen: set[int] = set()
    for text in texts:
        for match in PR_URL_RE.finditer(text):
            if match.group("repo").lower() != repo.lower():
                continue
            _append_number(numbers, seen, int(match.group("number")))
        for match in FIX_REF_RE.finditer(text):
            _append_number(numbers, seen, int(match.group("number")))
    return numbers


def find_fixed_version_reference(texts: list[str]) -> FixEvidence | None:
    for text in texts:
        match = FIXED_VERSION_RE.search(text)
        if match:
            return FixEvidence("fixed_version_reference", "", match.group(0).strip())
    return None


def find_maintainer_confirmation(comments: list[dict[str, Any]]) -> FixEvidence | None:
    for comment in comments:
        association = comment.get("author_association", "")
        body = comment.get("body", "")
        if association not in MAINTAINER_ASSOCIATIONS:
            continue
        if re.search(r"\b(fixed|resolved|patched|addressed)\b", body, re.I):
            return FixEvidence("maintainer_confirmation", comment.get("html_url", ""), _shorten(body))
    return None


def _issue_texts(issue: dict[str, Any], comments: list[dict[str, Any]]) -> list[str]:
    texts = [issue.get("title", ""), issue.get("body", "")]
    texts.extend(comment.get("body", "") for comment in comments)
    return [text for text in texts if text]


def _append_number(numbers: list[int], seen: set[int], number: int) -> None:
    if number in seen:
        return
    seen.add(number)
    numbers.append(number)


def _shorten(text: str, limit: int = 180) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."
