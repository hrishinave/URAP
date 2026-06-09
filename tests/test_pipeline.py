import unittest
import tempfile
from pathlib import Path

from llm_fp_mining.config import load_config
from llm_fp_mining.pipeline import build_candidate_row, collect_candidates


class FakeCollectionClient:
    def __init__(self):
        self.issue_fetches = 0
        self.search_fetches = 0

    def search_issues_page(self, query, per_page=100):
        self.search_fetches += 1
        return (
            [
                {
                    "number": 7,
                    "html_url": "https://github.com/vllm-project/vllm/issues/7",
                    "repository_url": "https://api.github.com/repos/vllm-project/vllm",
                    "title": "Speculative decoding precision drift",
                    "state": "closed",
                    "created_at": "2026-01-01T00:00:00Z",
                }
            ],
            1,
            {},
        )

    def paginate_from_headers(self, headers):
        return []

    def get_issue(self, repo, number):
        self.issue_fetches += 1
        return {
            "number": number,
            "html_url": f"https://github.com/{repo}/issues/{number}",
            "title": "Speculative decoding precision drift",
            "body": "Speculative decoding verifier changed precision accumulation. Fixed by #8.",
            "state": "closed",
            "created_at": "2026-01-01T00:00:00Z",
            "closed_at": "2026-01-03T00:00:00Z",
            "updated_at": "2026-01-03T00:00:00Z",
            "labels": [{"name": "bug"}],
        }

    def get_issue_comments(self, repo, number):
        return []

    def get_pull(self, repo, number):
        return {
            "html_url": f"https://github.com/{repo}/pull/{number}",
            "merged_at": "2026-01-02T00:00:00Z",
        }


class PipelineTests(unittest.TestCase):
    def test_collect_deduplicates_overlapping_queries(self):
        config = load_config()
        client = FakeCollectionClient()

        rows = collect_candidates(config, client, repos=["vllm-project/vllm"], max_queries=2)

        self.assertEqual(len(rows), 1)
        self.assertEqual(client.issue_fetches, 1)
        self.assertEqual(rows[0]["evidence_type"], "linked_merged_pr")
        self.assertEqual(rows[0]["novelty_status"], "candidate_new_trigger")

    def test_collect_resumes_completed_search_queries_from_cache(self):
        config = load_config()
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "search_cache.json"
            first_client = FakeCollectionClient()

            first_rows = collect_candidates(
                config,
                first_client,
                repos=["vllm-project/vllm"],
                max_queries=1,
                search_cache_path=cache_path,
            )

            second_client = FakeCollectionClient()
            second_rows = collect_candidates(
                config,
                second_client,
                repos=["vllm-project/vllm"],
                max_queries=1,
                search_cache_path=cache_path,
                resume_search=True,
            )

            self.assertEqual(len(first_rows), 1)
            self.assertEqual(len(second_rows), 1)
            self.assertEqual(first_client.search_fetches, 1)
            self.assertEqual(second_client.search_fetches, 0)

    def test_build_candidate_row_handles_null_github_text_fields(self):
        config = load_config()
        client = NullTextClient()
        match = {
            "repo": "vllm-project/vllm",
            "number": 99,
            "item": {
                "number": 99,
                "html_url": "https://github.com/vllm-project/vllm/issues/99",
                "repository_url": "https://api.github.com/repos/vllm-project/vllm",
                "title": None,
                "state": "closed",
            },
            "matched_terms": {"nondeterministic", "precision"},
            "matched_queries": {"query"},
        }

        row = build_candidate_row(config, client, match, include_without_fix=True)

        self.assertIsNotNone(row)
        self.assertEqual(row["title"], "")
        self.assertEqual(row["evidence_type"], "")


class NullTextClient:
    def get_issue(self, repo, number):
        return {
            "number": number,
            "html_url": f"https://github.com/{repo}/issues/{number}",
            "title": None,
            "body": None,
            "state": "closed",
            "created_at": None,
            "closed_at": None,
            "updated_at": None,
            "labels": [{"name": None}],
        }

    def get_issue_comments(self, repo, number):
        return [{"body": None, "author_association": "MEMBER"}]


if __name__ == "__main__":
    unittest.main()
