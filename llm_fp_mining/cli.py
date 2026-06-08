from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import DEFAULT_CONFIG_PATH, load_config, resolve_output_path
from .github import DEFAULT_PER_PAGE, GitHubClient
from .pipeline import collect_candidates, export_validated_cases, write_candidates_csv
from .queries import build_query_specs, build_search_query


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(message)s")

    config = load_config(args.config)
    if args.command == "collect":
        return run_collect(args, config)
    if args.command == "export-validated":
        return run_export_validated(args, config)
    parser.error(f"Unknown command: {args.command}")
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mine GitHub issues for LLM FP-order nondeterminism cases.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to mining JSON config.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="Collect candidate closed issues from GitHub.")
    collect.add_argument("--repo", action="append", help="Restrict to one repo. Repeat for multiple repos.")
    collect.add_argument("--output", help="Candidates CSV path. Defaults to config output.candidates_csv.")
    collect.add_argument("--max-queries", type=int, help="Limit generated queries for smoke tests.")
    collect.add_argument("--include-without-fix", action="store_true", help="Keep candidates even when no fix evidence is found.")
    collect.add_argument("--no-comments", action="store_true", help="Skip fetching issue comments.")
    collect.add_argument("--per-page", type=int, default=DEFAULT_PER_PAGE)
    collect.add_argument("--dry-run", action="store_true", help="Print generated queries without calling GitHub.")

    export = subparsers.add_parser("export-validated", help="Export manually labeled valid rows.")
    export.add_argument("--review-csv", help="Reviewed candidates CSV path. Defaults to config output.candidates_csv.")
    export.add_argument("--output", help="Validated CSV path. Defaults to config output.validated_csv.")
    return parser


def run_collect(args: argparse.Namespace, config) -> int:
    specs = build_query_specs(config, repos=args.repo)
    if args.max_queries is not None:
        specs = specs[: args.max_queries]

    if args.dry_run:
        for spec in specs:
            print(build_search_query(spec, config.start_date, config.end_date))
        print(f"Generated {len(specs)} queries.")
        return 0

    client = GitHubClient()
    rows = collect_candidates(
        config,
        client,
        repos=args.repo,
        max_queries=args.max_queries,
        include_without_fix=args.include_without_fix,
        fetch_comments=not args.no_comments,
        per_page=args.per_page,
    )
    output = resolve_output_path(config, args.output, config.candidates_csv)
    write_candidates_csv(rows, output)
    print(f"Wrote {len(rows)} candidate rows to {output}")
    return 0


def run_export_validated(args: argparse.Namespace, config) -> int:
    review_csv = resolve_output_path(config, args.review_csv, config.candidates_csv)
    output = resolve_output_path(config, args.output, config.validated_csv)
    rows = export_validated_cases(Path(review_csv), Path(output))
    print(f"Wrote {len(rows)} validated rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
