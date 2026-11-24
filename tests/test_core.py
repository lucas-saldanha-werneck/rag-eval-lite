import json
import os
import unittest

from rageval.core import BM25, check_grounded, retrieval_metrics, run, tokenize

ROOT = os.path.dirname(os.path.dirname(__file__))
CORPUS = os.path.join(ROOT, "data", "example_corpus.json")
QUESTIONS = os.path.join(ROOT, "data", "example_questions.json")
with open(CORPUS) as _f:
    CHUNKS = json.load(_f)["chunks"]


class Tests(unittest.TestCase):
    def test_tokenize_drops_stopwords(self):
        self.assertEqual(tokenize("How do I reset the Password?"), ["reset", "password"])

    def test_search_finds_exact_topic(self):
        self.assertEqual(BM25(CHUNKS).search("reset password", 1), ["c2"])

    def test_search_no_overlap_returns_empty(self):
        self.assertEqual(BM25(CHUNKS).search("zebra", 3), [])

    def test_metrics_perfect(self):
        qs = [{"question": "reset password", "gold": "c2"}]
        m = retrieval_metrics(BM25(CHUNKS), qs, 3)
        self.assertEqual(m["recall@3"], 1.0)
        self.assertEqual(m["mrr"], 1.0)

    def test_metrics_miss_is_reported(self):
        qs = [{"question": "zebra", "gold": "c2"}]
        m = retrieval_metrics(BM25(CHUNKS), qs, 3)
        self.assertEqual(m["recall@3"], 0.0)
        self.assertEqual(len(m["misses"]), 1)

    def test_grounded_ok(self):
        r = check_grounded("Open the login page and choose the forgot password link [c2].", CHUNKS)
        self.assertTrue(r["ok"])

    def test_missing_citation(self):
        r = check_grounded("Refunds are paid within 7 days.", CHUNKS)
        self.assertEqual(r["problems"][0]["issue"], "no citation")

    def test_unknown_chunk(self):
        r = check_grounded("Quizzes show answers [c9].", CHUNKS)
        self.assertIn("unknown chunk c9", r["problems"][0]["issue"])

    def test_weak_support(self):
        r = check_grounded("Premium members get free telescopes every month [c3].", CHUNKS)
        self.assertIn("weak support", r["problems"][0]["issue"])

    def test_example_run_is_stable(self):
        res = run(CORPUS, QUESTIONS, 3)
        self.assertEqual(res["retrieval"]["n"], 6)
        self.assertEqual([g["ok"] for g in res["grounding"]], [True, False, False])


if __name__ == "__main__":
    unittest.main()
