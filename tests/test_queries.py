from datetime import date
import unittest

from llm_fp_mining.config import load_config
from llm_fp_mining.queries import build_query_specs, build_search_query


class QueryTests(unittest.TestCase):
    def test_builds_paired_closed_issue_query(self):
        config = load_config()
        specs = build_query_specs(config, repos=["vllm-project/vllm"])

        query = build_search_query(specs[0], date(2025, 6, 1), date(2026, 6, 8))

        self.assertIn("repo:vllm-project/vllm", query)
        self.assertIn("is:issue", query)
        self.assertIn("is:closed", query)
        self.assertIn("created:2025-06-01..2026-06-08", query)
        self.assertIn('"nondeterministic"', query)

    def test_query_generation_is_deduplicated(self):
        config = load_config()
        specs = build_query_specs(config, repos=["vllm-project/vllm"])
        keys = [spec.query_key for spec in specs]

        self.assertEqual(len(keys), len(set(keys)))


if __name__ == "__main__":
    unittest.main()
