"""Vanilla RAG baseline.

Dense retrieval -> concatenate top-k chunks -> generate. No hybrid retrieval,
no reranking, no claim extraction, no conflict handling.

It shares the SAME index, embedder, and corpus as VeriRAG. That matters: if
the baseline used a different index, any measured difference would confound
"conflict handling helps" with "this index is better", and the comparison
would prove nothing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import config
from agents.generator import vanilla_generate
from agents.llm import LLMClient


@dataclass
class VanillaState:
    question: str
    retrieved_chunks: list = field(default_factory=list)
    answer: str = ""
    llm_calls: int = 0
    latency: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        # Emit an `evidence`-shaped list even though the baseline does no claim
        # extraction. The metrics read retrieval quality off this field, and
        # without it the baseline scored a retrieval hit rate of 0.000 -- a
        # measurement artifact, not a real failure, which would have flattered
        # VeriRAG in the comparison.
        pseudo_evidence = [
            {
                "claim_id": i,
                "chunk_id": c.chunk.chunk_id,
                "source": c.chunk.title,
                "supports_question": True,
            }
            for i, c in enumerate(self.retrieved_chunks)
        ]
        # Citations are parsed from the [n] markers the generator actually
        # emitted, not set to "every chunk retrieved". Listing all retrieved
        # chunks made the baseline score 0.963 on citation accuracy simply by
        # citing everything -- a trivially-won metric that made the comparison
        # meaningless.
        import re as _re

        cited_idx = {
            int(m) for m in _re.findall(r"\[(\d+)\]", self.answer or "")
        }
        citations = [
            c.chunk.title
            for i, c in enumerate(self.retrieved_chunks)
            if i in cited_idx
        ]
        return {
            "question": self.question,
            "answer": self.answer,
            "evidence": pseudo_evidence,
            "citations": citations,
            "retrieved": [c.chunk.chunk_id for c in self.retrieved_chunks],
            "llm_calls": self.llm_calls,
            "latency": round(self.latency, 3),
            "conflict_detected": False,
            "adjudicator_invoked": False,
        }


class VanillaRAG:
    def __init__(self, retriever, client: LLMClient, top_k: int | None = None):
        self.retriever = retriever
        self.client = client
        self.top_k = top_k or config.VANILLA_TOP_K

    def run(self, question: str) -> VanillaState:
        state = VanillaState(question=question)
        calls_before = self.client.call_count
        started = time.time()

        state.retrieved_chunks = self.retriever.dense_only(question, self.top_k)
        state.answer = vanilla_generate(
            self.client, question, state.retrieved_chunks
        )

        state.latency = time.time() - started
        state.llm_calls = self.client.call_count - calls_before
        return state

    def run_dict(self, question: str) -> dict[str, Any]:
        return self.run(question).to_dict()
