import csv
import tempfile
from pathlib import Path
import unittest

from llm_fp_mining.pipeline import CANDIDATE_FIELDS, export_validated_cases, write_candidates_csv


class ExportTests(unittest.TestCase):
    def test_exports_only_valid_rows_with_fix_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            review = tmp_path / "review.csv"
            output = tmp_path / "validated.csv"
            rows = [
                _row("1", "yes", "linked_merged_pr"),
                _row("2", "", "linked_merged_pr"),
                _row("3", "yes", ""),
            ]
            write_candidates_csv(rows, review)

            exported = export_validated_cases(review, output)

            self.assertEqual([row["issue_number"] for row in exported], ["1"])
            with output.open("r", encoding="utf-8", newline="") as f:
                csv_rows = list(csv.DictReader(f))
            self.assertEqual(len(csv_rows), 1)
            self.assertEqual(csv_rows[0]["issue_number"], "1")


def _row(number, is_valid, evidence_type):
    row = {field: "" for field in CANDIDATE_FIELDS}
    row.update(
        {
            "repo": "vllm-project/vllm",
            "issue_number": number,
            "issue_url": f"https://github.com/vllm-project/vllm/issues/{number}",
            "is_valid": is_valid,
            "evidence_type": evidence_type,
        }
    )
    return row


if __name__ == "__main__":
    unittest.main()
