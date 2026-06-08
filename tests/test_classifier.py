import unittest

from llm_fp_mining.classifier import classify_text
from llm_fp_mining.config import load_config


class ClassifierTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def test_known_prior_work_wins_over_candidate_trigger(self):
        text = "Chunked prefill and KV cache boundary changes caused different output at temperature 0."

        result = classify_text(text, self.config)

        self.assertEqual(result.novelty_status, "known_prior_work")
        self.assertIn("batch_size_load", result.prior_work_match)

    def test_candidate_new_trigger_for_speculative_decoding(self):
        text = "Speculative decoding verifier accepted tokens caused precision drift and inconsistent logits."

        result = classify_text(text, self.config)

        self.assertEqual(result.novelty_status, "candidate_new_trigger")
        self.assertIn("speculative_decoding", result.suspected_trigger)

    def test_invalid_without_fp_or_trigger_path(self):
        text = "The page renders inconsistently in the browser."

        result = classify_text(text, self.config)

        self.assertEqual(result.novelty_status, "invalid_or_no_fp_path")


if __name__ == "__main__":
    unittest.main()
