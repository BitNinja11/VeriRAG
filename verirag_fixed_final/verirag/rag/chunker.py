"""Split documents into overlapping chunks that carry their metadata forward."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .loader import Document


@dataclass
class Chunk:
    chunk_id: str
    text: str
    doc_id: str
    title: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def date(self) -> str:
        return self.metadata.get("date", "")

    @property
    def source_type(self) -> str:
        return self.metadata.get("source_type", "unknown")

    def citation(self) -> str:
        date = self.date
        return f"{self.title} ({date})" if date else self.title

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "doc_id": self.doc_id,
            "title": self.title,
            "date": self.date,
            "source_type": self.source_type,
        }


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_document(
    doc: Document, chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    """Paragraph-aware chunking with a word-count budget.

    Paragraphs are packed until the budget is hit rather than cutting mid
    sentence, because the Evidence Analyst has to quote spans and a chunk that
    severs "18 days of paid annual leave" is useless downstream.
    """
    paragraphs = _split_paragraphs(doc.text)
    chunks: list[Chunk] = []
    current: list[str] = []
    current_words = 0

    def flush() -> None:
        nonlocal current, current_words
        if not current:
            return
        text = "\n\n".join(current)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::c{len(chunks)}",
                text=text,
                doc_id=doc.doc_id,
                title=doc.title,
                metadata=dict(doc.metadata),
            )
        )
        # Carry the tail of this chunk into the next for overlap.
        if chunk_overlap > 0 and current:
            tail_words: list[str] = []
            for para in reversed(current):
                words = para.split()
                if len(tail_words) + len(words) > chunk_overlap:
                    break
                tail_words = words + tail_words
            carry = " ".join(tail_words)
            current = [carry] if carry else []
            current_words = len(tail_words)
        else:
            current = []
            current_words = 0

    for para in paragraphs:
        words = len(para.split())

        # A single oversized paragraph becomes its own chunk.
        if words > chunk_size:
            flush()
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::c{len(chunks)}",
                    text=para,
                    doc_id=doc.doc_id,
                    title=doc.title,
                    metadata=dict(doc.metadata),
                )
            )
            current, current_words = [], 0
            continue

        if current_words + words > chunk_size:
            flush()

        current.append(para)
        current_words += words

    flush()
    return chunks


def chunk_documents(
    docs: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, chunk_size, chunk_overlap))
    return chunks
