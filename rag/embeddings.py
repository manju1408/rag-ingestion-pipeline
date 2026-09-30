"""Embedding backends. The default hashing embedder is deterministic and offline,
which keeps tests and CI reproducible; swap in a sentence-transformer in production."""
from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Signed feature hashing over unigrams + bigrams, L2-normalized."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _bucket(self, token: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "little")
        return h % self.dim, 1.0 if (h >> 63) & 1 else -1.0

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            toks = TOKEN.findall(text.lower())
            for tok in toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]:
                idx, sign = self._bucket(tok)
                out[row, idx] += sign
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


class SentenceTransformerEmbedder:  # pragma: no cover - optional dependency
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._m = SentenceTransformer(model)
        self.dim = self._m.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._m.encode(texts, normalize_embeddings=True), dtype=np.float32)
