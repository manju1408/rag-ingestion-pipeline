"""Heading-aware recursive chunking with overlap."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .ingest import Document

SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    section: str
    position: int


def _split(text: str, size: int, seps: list[str]) -> list[str]:
    if len(text) <= size:
        return [text]
    if not seps:
        return [text[i : i + size] for i in range(0, len(text), size)]
    sep, rest = seps[0], seps[1:]
    parts, out, buf = text.split(sep), [], ""
    for part in parts:
        candidate = f"{buf}{sep}{part}" if buf else part
        if len(candidate) <= size:
            buf = candidate
            continue
        if buf:
            out.append(buf)
        buf = ""
        if len(part) > size:
            out.extend(_split(part, size, rest))
        else:
            buf = part
    if buf:
        out.append(buf)
    return out


def _sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, body) pairs so chunks never straddle sections."""
    pieces = re.split(r"(?m)^(#{1,6} .+)$", text)
    sections, heading = [], ""
    if pieces[0].strip():
        sections.append(("", pieces[0].strip()))
    for i in range(1, len(pieces), 2):
        heading = pieces[i].lstrip("# ").strip()
        body = pieces[i + 1].strip() if i + 1 < len(pieces) else ""
        if body:
            sections.append((heading, body))
    return sections


def chunk_document(doc: Document, size: int = 800, overlap: int = 120) -> list[Chunk]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    chunks: list[Chunk] = []
    for heading, body in _sections(doc.text):
        pieces = _split(body, size - overlap, SEPARATORS)
        for i, piece in enumerate(pieces):
            tail = pieces[i - 1][-overlap:] if i and overlap else ""
            text = (tail + " " + piece).strip() if tail else piece
            n = len(chunks)
            chunks.append(Chunk(f"{doc.doc_id}-{n:04d}", doc.doc_id, text, heading, n))
    return chunks
