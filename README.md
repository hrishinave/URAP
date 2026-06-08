# LLM FP-Order Nondeterminism Mining

This repo now has two workflows:

- `app.py`: the original Flask search UI for quick manual GitHub issue lookup.
- `llm_fp_mining`: a reproducible CLI pipeline for mining closed LLM runtime issues from GitHub.

The CLI targets closed issues from June 1, 2025 through June 8, 2026 and tags cases that match known prior work from the Thinking Machines batch-invariance blog or the TP/all-reduce paper.

## Dry Run Queries

```bash
python -m llm_fp_mining.cli collect --dry-run --max-queries 10
```

## Collect Candidates

Set a GitHub token first:

```bash
export GITHUB_TOKEN=...
python -m llm_fp_mining.cli collect --max-queries 25
```

The default output is `data/candidates.csv`. For a small smoke run:

```bash
python -m llm_fp_mining.cli collect --repo vllm-project/vllm --max-queries 5 --include-without-fix
```

## Manual Review

Edit `data/candidates.csv` and fill:

- `is_valid`
- `novelty_status`
- `trigger_category`
- `fp_order_mechanism`
- `fix_summary`
- `invalid_reason`

Then export validated rows:

```bash
python -m llm_fp_mining.cli export-validated
```

Rows are exported to `data/validated_cases.csv` only when `is_valid` is true and fix evidence is present.
