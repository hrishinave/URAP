import unittest

from llm_fp_mining.evidence import extract_pr_numbers, find_fix_evidence


class FakeClient:
    def get_pull(self, repo, number):
        return {
            "html_url": f"https://github.com/{repo}/pull/{number}",
            "merged_at": "2026-01-02T03:04:05Z",
        }


class EvidenceTests(unittest.TestCase):
    def test_extracts_fix_pr_refs(self):
        refs = extract_pr_numbers(
            "vllm-project/vllm",
            [
                "Fixed by #123",
                "See https://github.com/vllm-project/vllm/pull/456",
                "Unrelated https://github.com/other/repo/pull/999",
            ],
        )

        self.assertEqual(refs, [123, 456])

    def test_linked_merged_pr_is_fix_evidence(self):
        issue = {"title": "nondeterministic fp8 output", "body": "Fixed by #123"}

        evidence = find_fix_evidence("vllm-project/vllm", issue, [], client=FakeClient())

        self.assertEqual(evidence.evidence_type, "linked_merged_pr")
        self.assertEqual(evidence.linked_fix_url, "https://github.com/vllm-project/vllm/pull/123")

    def test_maintainer_confirmation_is_fix_evidence(self):
        comments = [
            {
                "author_association": "MEMBER",
                "body": "This is fixed in the latest release.",
                "html_url": "https://github.com/repo/issues/1#comment",
            }
        ]

        evidence = find_fix_evidence("repo/project", {"title": "", "body": ""}, comments)

        self.assertEqual(evidence.evidence_type, "fixed_version_reference")


if __name__ == "__main__":
    unittest.main()
