"""VeriRAG pipeline orchestration.

Flow:

    retrieve (hybrid) -> rerank -> extract evidence -> detect conflict
         -> [conditional] adjudicate -> generate

The adjudicator executes only when a CONTRADICTION exists, so questions whose
sources agree cost one fewer model call. `adjudicator_invoked` is recorded per
query so the evaluation can report how often that path fires.

This is implemented as an explicit state machine rather than with LangGraph.
The graph has six nodes and one branch; hand-rolling it keeps the dependency
list short and makes the control flow readable in one screen.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import config
from agents.adjudicator import Adjudication, Adjudicator, evidence_scores
from agents.conflict_detector import (
    ConflictDetector,
    ConflictRelation,
    has_real_conflict,
)
from agents.evidence_analyst import Evidence, EvidenceAnalyst
from agents.generator import AnswerGenerator, GeneratedAnswer
from agents.llm import LLMClient
from rag.chunker import chunk_documents
from rag.embeddings import EmbeddingBackend
from rag.hybrid_retriever import HybridRetriever
from rag.loader import load_documents
from rag.reranker import Reranker


@dataclass
class VeriRAGState:
    """Shared state threaded through every node."""

    question: str
    retrieved_chunks: list = field(default_factory=list)
    reranked_chunks: list = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    conflicts: list[ConflictRelation] = field(default_factory=list)
    adjudication: Adjudication | None = None
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    confidence: str = "medium"
    adjudicator_invoked: bool = False
    llm_calls: int = 0
    latency: float = 0.0
    stage_timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": self.citations,
            "confidence": self.confidence,
            "conflict_detected": has_real_conflict(self.conflicts),
            "adjudicator_invoked": self.adjudicator_invoked,
            "evidence": [e.to_dict() for e in self.evidence],
            "conflicts": [c.to_dict() for c in self.conflicts],
            "adjudication": (
                self.adjudication.to_dict() if self.adjudication else None
            ),
            "llm_calls": self.llm_calls,
            "latency": round(self.latency, 3),
            "stage_timings": {
                k: round(v, 3) for k, v in self.stage_timings.items()
            },
        }


class VeriRAG:
    def __init__(
        self,
        documents_dir: str | None = None,
        client: LLMClient | None = None,
        verbose: bool = False,
    ):
        self.verbose = verbose
        self.client = client or LLMClient()

        documents_dir = documents_dir or config.DOCUMENTS_DIR
        self.documents = load_documents(documents_dir)
        self.chunks = chunk_documents(
            self.documents, config.CHUNK_SIZE, config.CHUNK_OVERLAP
        )

        self.embedder = EmbeddingBackend(config.EMBEDDING_MODEL)
        self.retriever = HybridRetriever(
            self.chunks,
            self.embedder,
            dense_top_k=config.DENSE_TOP_K,
            bm25_top_k=config.BM25_TOP_K,
            hybrid_top_k=config.HYBRID_TOP_K,
            rrf_k=config.RRF_K,
        )
        self.reranker = Reranker(config.RERANKER_MODEL)

        self.analyst = EvidenceAnalyst(self.client)
        self.detector = ConflictDetector(self.client)
        self.adjudicator = Adjudicator(self.client)
        self.generator = AnswerGenerator(self.client)

    # -- node functions ----------------------------------------------------

    def _node_retrieve(self, state: VeriRAGState) -> None:
        state.retrieved_chunks = self.retriever.retrieve(state.question)

    def _node_rerank(self, state: VeriRAGState) -> None:
        state.reranked_chunks = self.reranker.rerank(
            state.question, state.retrieved_chunks, config.RERANK_TOP_K
        )

    def _node_extract_evidence(self, state: VeriRAGState) -> None:
        state.evidence = self.analyst.analyse(
            state.question, state.reranked_chunks
        )

    def _node_detect_conflict(self, state: VeriRAGState) -> None:
        state.conflicts = self.detector.detect(state.question, state.evidence)

    def _node_adjudicate(self, state: VeriRAGState) -> None:
        state.adjudication = self.adjudicator.adjudicate(
            state.question, state.evidence, state.conflicts
        )
        state.adjudicator_invoked = True

    def _node_generate(self, state: VeriRAGState) -> None:
        result: GeneratedAnswer = self.generator.generate(
            state.question, state.evidence, state.conflicts, state.adjudication
        )
        state.answer = result.answer
        state.citations = result.citations
        state.confidence = result.confidence

    # -- orchestration -----------------------------------------------------

    def run(self, question: str) -> VeriRAGState:
        state = VeriRAGState(question=question)
        calls_before = self.client.call_count
        started = time.time()

        stages = [
            ("retrieve", self._node_retrieve),
            ("rerank", self._node_rerank),
            ("extract_evidence", self._node_extract_evidence),
            ("detect_conflict", self._node_detect_conflict),
        ]

        for name, fn in stages:
            t0 = time.time()
            fn(state)
            state.stage_timings[name] = time.time() - t0
            if self.verbose:
                print(f"  [{name}] {state.stage_timings[name]:.3f}s")

        # Conditional edge: adjudicate only when a contradiction exists.
        if has_real_conflict(state.conflicts):
            t0 = time.time()
            self._node_adjudicate(state)
            state.stage_timings["adjudicate"] = time.time() - t0
            if self.verbose:
                print(f"  [adjudicate] {state.stage_timings['adjudicate']:.3f}s")
        elif self.verbose:
            print("  [adjudicate] skipped (no contradiction)")

        t0 = time.time()
        self._node_generate(state)
        state.stage_timings["generate"] = time.time() - t0

        state.latency = time.time() - started
        state.llm_calls = self.client.call_count - calls_before
        return state

    def run_dict(self, question: str) -> dict[str, Any]:
        return self.run(question).to_dict()

    def evidence_scores(self, state: VeriRAGState) -> dict[int, float]:
        return evidence_scores(state.evidence)
