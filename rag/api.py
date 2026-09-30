"""FastAPI service: POST /query returns top chunks and a grounded prompt.

Run: RAG_DOCS=data/docs RAG_API_KEY=dev uvicorn rag.api:app --reload
"""
from __future__ import annotations

import os
import secrets

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .pipeline import build_index, build_prompt
from .store import VectorStore


class Query(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    k: int = Field(default=4, ge=1, le=20)


class Hit(BaseModel):
    chunk_id: str
    section: str
    score: float
    text: str


class Answer(BaseModel):
    hits: list[Hit]
    prompt: str


def create_app(store: VectorStore, api_key: str) -> FastAPI:
    app = FastAPI(title="RAG retrieval service")

    def auth(x_api_key: str = Header(default="")):
        if not secrets.compare_digest(x_api_key, api_key):
            raise HTTPException(status_code=401, detail="invalid api key")

    @app.get("/health")
    def health():
        return {"status": "ok", "chunks": len(store)}

    @app.post("/query", response_model=Answer, dependencies=[Depends(auth)])
    def query(q: Query):
        hits = store.search(q.question, k=q.k)
        return Answer(
            hits=[Hit(chunk_id=c.chunk_id, section=c.section, score=round(s, 4), text=c.text) for c, s in hits],
            prompt=build_prompt(q.question, hits),
        )

    return app


if os.getenv("RAG_DOCS"):  # pragma: no cover
    app = create_app(build_index(os.environ["RAG_DOCS"]), os.environ.get("RAG_API_KEY", ""))
