"""Expose the store as a LangChain retriever (requires ``langchain-core``)."""
from __future__ import annotations

from .store import VectorStore


def as_retriever(store: VectorStore, k: int = 4):  # pragma: no cover - optional dependency
    from langchain_core.documents import Document as LCDocument
    from langchain_core.retrievers import BaseRetriever

    class _Retriever(BaseRetriever):
        def _get_relevant_documents(self, query, *, run_manager=None):
            return [
                LCDocument(page_content=c.text,
                           metadata={"chunk_id": c.chunk_id, "doc_id": c.doc_id,
                                     "section": c.section, "score": s})
                for c, s in store.search(query, k=k)
            ]

    return _Retriever()
