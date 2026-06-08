from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

from .config import MiningConfig


@dataclass(frozen=True)
class SearchQuerySpec:
    repo: str
    terms: tuple[str, ...]

    @property
    def query_key(self) -> str:
        return "|".join((self.repo, *self.terms))


def quote_term(term: str) -> str:
    escaped = term.replace('"', '\\"')
    return f'"{escaped}"'


def build_query_specs(config: MiningConfig, repos: Iterable[str] | None = None) -> list[SearchQuerySpec]:
    selected_repos = tuple(repos) if repos is not None else config.repos
    strategy = config.query_strategy
    if strategy != "paired":
        raise ValueError(f"Unsupported query_strategy: {strategy}")

    nondet_terms = config.keyword_groups.get("nondeterminism_terms", ())
    fp_terms = config.keyword_groups.get("fp_order_terms", ())
    trigger_terms = config.keyword_groups.get("trigger_terms", ())

    specs: list[SearchQuerySpec] = []
    seen: set[str] = set()
    for repo in selected_repos:
        for nondet in nondet_terms:
            for paired_term in (*fp_terms, *trigger_terms):
                spec = SearchQuerySpec(repo=repo, terms=(nondet, paired_term))
                if spec.query_key in seen:
                    continue
                seen.add(spec.query_key)
                specs.append(spec)
    return specs


def build_search_query(spec: SearchQuerySpec, start_date: date, end_date: date) -> str:
    term_clause = " ".join(quote_term(term) for term in spec.terms)
    return (
        f"{term_clause} repo:{spec.repo} is:issue is:closed "
        f"created:{start_date.isoformat()}..{end_date.isoformat()}"
    )
