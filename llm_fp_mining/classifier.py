from __future__ import annotations

from dataclasses import dataclass

from .config import MiningConfig


@dataclass(frozen=True)
class Classification:
    matched_terms: tuple[str, ...]
    suspected_trigger: tuple[str, ...]
    prior_work_match: tuple[str, ...]
    novelty_status: str


def classify_text(text: str, config: MiningConfig) -> Classification:
    matched_terms = []
    for terms in config.keyword_groups.values():
        matched_terms.extend(find_terms(text, terms))

    prior_work_match = []
    for category, terms in config.known_prior_work.items():
        if find_terms(text, terms):
            prior_work_match.append(category)

    suspected_trigger = []
    for category, terms in config.candidate_triggers.items():
        if find_terms(text, terms):
            suspected_trigger.append(category)

    if prior_work_match:
        novelty_status = "known_prior_work"
    elif suspected_trigger:
        novelty_status = "candidate_new_trigger"
    elif find_terms(text, config.keyword_groups.get("fp_order_terms", ())):
        novelty_status = "needs_review"
    else:
        novelty_status = "invalid_or_no_fp_path"

    return Classification(
        matched_terms=tuple(sorted(set(matched_terms), key=str.lower)),
        suspected_trigger=tuple(suspected_trigger),
        prior_work_match=tuple(prior_work_match),
        novelty_status=novelty_status,
    )


def find_terms(text: str, terms: tuple[str, ...]) -> tuple[str, ...]:
    normalized = normalize(text)
    matches = []
    for term in terms:
        if normalize(term) in normalized:
            matches.append(term)
    return tuple(matches)


def normalize(text: str | None) -> str:
    return " ".join((text or "").lower().replace("_", " ").replace("-", " ").split())
