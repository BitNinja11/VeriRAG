"""Regression tests for bugs found during the VeriRAG audit.

Run with:
    python -m unittest discover -s tests -v

These tests intentionally use only the deterministic/offline backend so CI does
not require API keys, model downloads, or network access.
"""

from __future__ import annotations

import unittest

import config
from agents.adjudicator import deterministic_adjudicate
from agents.conflict_detector import ConflictRelation, deterministic_detect
from agents.evidence_analyst import Evidence, deterministic_extract
from agents.llm import LLMClient
from evaluation.evaluate import DISTRACTOR_VALUES, load_benchmark
from evaluation.metrics import aggregate, answer_is_correct, score_item
from pipeline.verirag import VeriRAG
from rag.embeddings import EmbeddingBackend


class EvidenceExtractionRegressionTests(unittest.TestCase):
    def test_remote_work_is_not_misread_as_leave_accrual(self):
        result = deterministic_extract(
            "How many days per month may employees work remotely?",
            "Employees may work remotely up to 8 days per month with manager approval.",
            {"title": "Employee Handbook", "source_type": "handbook"},
        )
        self.assertEqual(result["property"], "remote_work_days")
        self.assertEqual(result["value"], "8 days")
        self.assertEqual(result["numeric_value"], 8.0)
        self.assertTrue(result["supports_question"])

    def test_refund_processing_preserves_range_and_property(self):
        result = deterministic_extract(
            "How many days does refund processing take?",
            "Customers may request a refund within 30 days. Processing takes 3 to 5 business days.",
            {"title": "Refund Policy", "source_type": "official_policy"},
        )
        self.assertEqual(result["property"], "refund_processing_days")
        self.assertEqual(result["value"], "3 to 5 business days")
        self.assertIsNone(result["numeric_value"])
        self.assertTrue(result["supports_question"])

    def test_password_compound_sentence_selects_rotation_quantity(self):
        text = "Passwords must be at least 14 characters and must be rotated every 180 days."
        result = deterministic_extract(
            "How often must passwords be rotated?",
            text,
            {"title": "Security Policy", "source_type": "official_policy"},
        )
        self.assertEqual(result["property"], "password_rotation_days")
        self.assertEqual(result["value"], "180 days")
        self.assertEqual(result["numeric_value"], 180.0)

    def test_password_compound_sentence_selects_length_quantity(self):
        text = "Passwords must be at least 14 characters and must be rotated every 180 days."
        result = deterministic_extract(
            "What is the minimum password length?",
            text,
            {"title": "Security Policy", "source_type": "official_policy"},
        )
        self.assertEqual(result["property"], "password_min_length")
        self.assertEqual(result["value"], "14 characters")
        self.assertEqual(result["numeric_value"], 14.0)

    def test_categorical_schedule_answer_is_supported(self):
        result = deterministic_extract(
            "On which day is building maintenance scheduled?",
            "Building maintenance is scheduled for the first Saturday of every month.",
            {"title": "Facilities Guide", "source_type": "handbook"},
        )
        self.assertEqual(result["property"], "maintenance_schedule_day")
        self.assertEqual(result["value"], "first Saturday")
        self.assertTrue(result["supports_question"])

    def test_active_and_passive_supersession_are_not_confused(self):
        question = "How many unused leave days can be carried forward?"
        body = "Employees may carry forward up to 10 unused leave days."

        active = deterministic_extract(
            question,
            body,
            {
                "title": "Current Policy",
                "source_type": "official_policy",
                "authority_note": "This policy supersedes the 2023 revision.",
            },
        )
        passive = deterministic_extract(
            question,
            body,
            {
                "title": "Old Policy",
                "source_type": "official_policy",
                "authority_note": "Superseded by the 2026 revision.",
            },
        )
        self.assertTrue(active["supersedes_previous"])
        self.assertFalse(passive["supersedes_previous"])


class AdjudicationRegressionTests(unittest.TestCase):
    @staticmethod
    def _evidence(claim_id: int, doc: str, value: str, supersedes: bool = False) -> Evidence:
        return Evidence(
            claim_id=claim_id,
            chunk_id=f"{doc}::c0",
            source=doc,
            source_type="official_policy",
            date="2026-01-01" if claim_id != 2 else "2024-01-01",
            claim=f"Policy states {value}.",
            value=value,
            property="refund_processing_days",
            scope="customers",
            supports_question=True,
            supersedes_previous=supersedes,
            quote=f"Policy states {value}.",
            confidence=0.9,
            relevance=0.9,
            numeric_value=None,
        )

    def test_agreeing_claim_is_not_rejected_with_conflicting_third_source(self):
        # Claims 0 and 1 agree; claim 2 contradicts both. Selecting 0 must not
        # incorrectly mark claim 1 as rejected just because all three lie in
        # one connected conflict component.
        evidence = [
            self._evidence(0, "current_policy", "3 to 5 business days", True),
            self._evidence(1, "current_policy_copy", "3 to 5 business days"),
            self._evidence(2, "old_policy", "5 to 7 business days"),
        ]
        relations = [
            ConflictRelation([0, 1], "SUPPORT", "same value", "rules"),
            ConflictRelation([0, 2], "CONTRADICTION", "different range", "rules"),
            ConflictRelation([1, 2], "CONTRADICTION", "different range", "rules"),
        ]
        result = deterministic_adjudicate(evidence, {0, 1, 2}, relations)
        self.assertEqual(result.selected_claim_id, 0)
        self.assertEqual(result.rejected_claim_ids, [2])

    def test_equivalent_range_formats_and_scope_spelling_support_each_other(self):
        left = self._evidence(0, "policy_a", "3 to 5 business days")
        right = self._evidence(1, "policy_b", "3–5 business days")
        left.scope = "full-time employees"
        right.scope = "full time employees"
        relations = deterministic_detect([left, right])
        self.assertEqual(len(relations), 1)
        self.assertEqual(relations[0].relationship, "SUPPORT")

    def test_low_authority_source_cannot_self_declare_supersession(self):
        official = self._evidence(0, "official", "30 days")
        blog = self._evidence(1, "blog", "90 days", True)
        blog.source_type = "blog"
        blog.date = "2026-05-20"
        relations = [
            ConflictRelation([0, 1], "CONTRADICTION", "different value", "rules")
        ]
        result = deterministic_adjudicate([official, blog], {0, 1}, relations)
        self.assertEqual(result.selected_claim_id, 0)
        self.assertEqual(result.criterion, "source_authority")



class PortabilityRegressionTests(unittest.TestCase):
    def test_tiny_tfidf_corpus_does_not_require_svd(self):
        backend = EmbeddingBackend("unused", prefer_transformer=False)
        matrix = backend.fit(["onlytoken"])
        query = backend.encode(["onlytoken"])
        self.assertEqual(matrix.shape[0], 1)
        self.assertEqual(query.shape[0], 1)
        self.assertEqual(matrix.shape[1], query.shape[1])


class EvaluationMetricRegressionTests(unittest.TestCase):
    def test_never_abstaining_does_not_get_credit_on_unanswerable_items(self):
        rows = [
            {
                "answer_correct": True,
                "expect_abstain": False,
                "abstained": False,
                "abstention_correct": True,
                "expected_conflict": False,
                "detected_conflict": False,
                "conflict_correct": True,
                "citation_correct": None,
                "primary_citation_correct": None,
                "retrieval_hit": None,
                "hallucinated_value": False,
                "adjudicator_invoked": False,
                "llm_calls": 0,
                "latency": 0.0,
            },
            {
                "answer_correct": False,
                "expect_abstain": True,
                "abstained": False,
                "abstention_correct": False,
                "expected_conflict": False,
                "detected_conflict": False,
                "conflict_correct": True,
                "citation_correct": None,
                "primary_citation_correct": None,
                "retrieval_hit": None,
                "hallucinated_value": False,
                "adjudicator_invoked": False,
                "llm_calls": 0,
                "latency": 0.0,
            },
        ]
        metrics = aggregate(rows)
        self.assertEqual(metrics["abstention_recall"], 0.0)
        self.assertEqual(metrics["false_abstention_rate"], 0.0)

    def test_range_answer_requires_the_whole_range(self):
        expected = ["3 to 5 business days"]
        self.assertTrue(
            answer_is_correct("Processing takes 3 to 5 business days.", expected)
        )
        self.assertTrue(
            answer_is_correct("Processing takes between 3 and 5 business days.", expected)
        )
        self.assertFalse(
            answer_is_correct("Processing takes 5 to 7 business days.", expected)
        )
        self.assertFalse(answer_is_correct("Processing takes 5 business days.", expected))

    def test_gold_hit_does_not_hide_wrong_primary_citation(self):
        item = {
            "id": "citation-test",
            "category": "clean",
            "question": "What is the value?",
            "expected_answer_any": ["24 days"],
            "expected_conflict": False,
            "gold_sources": ["gold_doc"],
        }
        result = {
            "answer": "The value is 24 days. [0]",
            "citations": ["[0] Wrong Source", "[1] Gold Source"],
            "evidence": [
                {"claim_id": 0, "chunk_id": "wrong_doc::c0", "source": "Wrong Source"},
                {"claim_id": 1, "chunk_id": "gold_doc::c0", "source": "Gold Source"},
            ],
        }
        row = score_item(item, result, [])
        self.assertTrue(row["citation_correct"])
        self.assertFalse(row["primary_citation_correct"])



class FullBenchmarkRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system = VeriRAG(client=LLMClient(provider="offline"))
        cls.questions = load_benchmark(config.BENCHMARK_PATH)

    def test_full_offline_benchmark_has_no_known_regressions(self):
        rows = []
        for item in self.questions:
            result = self.system.run_dict(item["question"])
            rows.append(score_item(item, result, DISTRACTOR_VALUES))
        metrics = aggregate(rows)

        self.assertEqual(metrics["answer_accuracy"], 1.0)
        self.assertEqual(metrics["abstention_recall"], 1.0)
        self.assertEqual(metrics["false_abstention_rate"], 0.0)
        self.assertEqual(metrics["conflict_accuracy"], 1.0)
        self.assertEqual(metrics["conflict_precision"], 1.0)
        self.assertEqual(metrics["conflict_recall"], 1.0)
        self.assertEqual(metrics["primary_citation_accuracy"], 1.0)
        self.assertEqual(metrics["hallucination_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
