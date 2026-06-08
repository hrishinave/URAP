from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "llm_fp_mining.json"


@dataclass(frozen=True)
class MiningConfig:
    start_date: date
    end_date: date
    repos: tuple[str, ...]
    query_strategy: str
    keyword_groups: dict[str, tuple[str, ...]]
    known_prior_work: dict[str, tuple[str, ...]]
    candidate_triggers: dict[str, tuple[str, ...]]
    candidates_csv: Path
    validated_csv: Path
    source_path: Path


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> MiningConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    start_date = date.fromisoformat(raw["date_range"]["start"])
    end_date = date.fromisoformat(raw["date_range"]["end"])
    if start_date > end_date:
        raise ValueError("date_range.start must be on or before date_range.end")

    output = raw.get("output", {})
    return MiningConfig(
        start_date=start_date,
        end_date=end_date,
        repos=tuple(raw["repos"]),
        query_strategy=raw.get("query_strategy", "paired"),
        keyword_groups=_tuple_dict(raw.get("keyword_groups", {})),
        known_prior_work=_tuple_dict(raw.get("known_prior_work", {})),
        candidate_triggers=_tuple_dict(raw.get("candidate_triggers", {})),
        candidates_csv=Path(output.get("candidates_csv", "data/candidates.csv")),
        validated_csv=Path(output.get("validated_csv", "data/validated_cases.csv")),
        source_path=config_path,
    )


def _tuple_dict(raw: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    return {key: tuple(str(term) for term in terms) for key, terms in raw.items()}


def resolve_output_path(config: MiningConfig, path: str | Path | None, default_path: Path) -> Path:
    selected = Path(path) if path is not None else default_path
    if selected.is_absolute():
        return selected
    return Path.cwd() / selected
