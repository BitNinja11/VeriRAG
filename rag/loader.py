"""Load documents and parse their metadata frontmatter."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def date(self) -> str:
        return self.metadata.get("date", "")

    @property
    def source_type(self) -> str:
        return self.metadata.get("source_type", "unknown")

    @property
    def version(self) -> int:
        try:
            return int(self.metadata.get("version", 0))
        except (TypeError, ValueError):
            return 0


_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    """Split a simple `key: value` frontmatter block from the body.

    Deliberately not using PyYAML: the format here is flat key/value only, and
    avoiding the dependency keeps the install light.
    """
    match = _FRONTMATTER.match(raw)
    if not match:
        return {}, raw

    meta: dict[str, Any] = {}
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()

    return meta, raw[match.end():]


def load_documents(directory: str) -> list[Document]:
    """Load every .txt file in `directory` as a Document, sorted by filename."""
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Document directory not found: {directory}")

    docs: list[Document] = []
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(directory, filename)
        with open(path, encoding="utf-8") as handle:
            raw = handle.read()

        meta, body = parse_frontmatter(raw)
        doc_id = os.path.splitext(filename)[0]
        title = meta.get("title", doc_id.replace("_", " ").title())

        docs.append(
            Document(
                doc_id=doc_id,
                title=title,
                text=body.strip(),
                metadata={**meta, "filename": filename},
            )
        )

    if not docs:
        raise ValueError(f"No .txt documents found in {directory}")

    return docs
