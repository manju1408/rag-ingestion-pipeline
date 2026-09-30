"""A small in-memory vector store with idempotent upserts and on-disk persistence."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .chunking import Chunk
from .embeddings import Embedder


class VectorStore:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.vectors = np.zeros((0, embedder.dim), dtype=np.float32)
        self._ids: set[str] = set()

    def __len__(self) -> int:
        return len(self.chunks)

    def upsert(self, chunks: list[Chunk]) -> int:
        """Add chunks not already indexed. Returns how many were new."""
        new = [c for c in chunks if c.chunk_id not in self._ids]
        if not new:
            return 0
        self.vectors = np.vstack([self.vectors, self.embedder.embed([c.text for c in new])])
        self.chunks.extend(new)
        self._ids.update(c.chunk_id for c in new)
        return len(new)

    def search(self, query: str, k: int = 4, min_score: float = 0.0) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        q = self.embedder.embed([query])[0]
        scores = self.vectors @ q
        order = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in order if scores[i] > min_score]

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        np.save(p / "vectors.npy", self.vectors)
        (p / "chunks.json").write_text(json.dumps([asdict(c) for c in self.chunks]))

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder) -> "VectorStore":
        p = Path(path)
        store = cls(embedder)
        store.vectors = np.load(p / "vectors.npy")
        store.chunks = [Chunk(**c) for c in json.loads((p / "chunks.json").read_text())]
        store._ids = {c.chunk_id for c in store.chunks}
        return store
