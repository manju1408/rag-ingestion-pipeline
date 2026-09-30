"""End-to-end ingestion and a grounded answer builder."""
from __future__ import annotations

from pathlib import Path

from .chunking import chunk_document
from .embeddings import Embedder, HashingEmbedder
from .ingest import load_documents
from .store import VectorStore


def build_index(docs_dir: str | Path, embedder: Embedder | None = None,
                store: VectorStore | None = None) -> VectorStore:
    store = store or VectorStore(embedder or HashingEmbedder())
    for doc in load_documents(docs_dir):
        store.upsert(chunk_document(doc))
    return store


def build_prompt(question: str, hits) -> str:
    """Prompt for an LLM: answer only from the numbered context, cite sources."""
    context = "\n\n".join(f"[{i + 1}] ({c.section or 'intro'}) {c.text}" for i, (c, _) in enumerate(hits))
    return (
        "Answer the question using only the context below. Cite sources like [1]. "
        "If the context does not contain the answer, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
