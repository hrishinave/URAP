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

    def test_triton_autotuner_is_not_prior_work(self):
        text = "[Bug]: vLLM produces non-deterministic output due to Triton autotuner."

        result = classify_text(text, self.config)

        self.assertEqual(result.novelty_status, "candidate_new_trigger")
        self.assertIn("cross_backend_architecture", result.suspected_trigger)
        self.assertEqual(result.prior_work_match, ())

    def test_tensor_parallel_all_reduce_is_prior_work(self):
        text = "Changing tensor parallel size changes all-reduce order and gives different output."

        result = classify_text(text, self.config)

        self.assertEqual(result.novelty_status, "known_prior_work")
        self.assertIn("tensor_parallel_all_reduce", result.prior_work_match)

    def test_invalid_without_fp_or_trigger_path(self):
        text = "The page renders inconsistently in the browser."

        result = classify_text(text, self.config)

        self.assertEqual(result.novelty_status, "invalid_or_no_fp_path")


if __name__ == "__main__":
    unittest.main()
