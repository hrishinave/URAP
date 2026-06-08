import unittest

from llm_fp_mining.config import load_config
from llm_fp_mining.pipeline import collect_candidates


class FakeCollectionClient:
    def __init__(self):
        self.issue_fetches = 0

    def search_issues_page(self, query, per_page=100):
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


if __name__ == "__main__":
    unittest.main()
