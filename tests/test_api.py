from fastapi.testclient import TestClient

from generate_docs import main as generate
from rag.api import create_app
from rag.pipeline import build_index


def client(tmp_path):
    generate(str(tmp_path))
    return TestClient(create_app(build_index(tmp_path), api_key="test-key"))


def test_requires_api_key(tmp_path):
    r = client(tmp_path).post("/query", json={"question": "refund policy"})
    assert r.status_code == 401


def test_query_returns_hits(tmp_path):
    r = client(tmp_path).post("/query", json={"question": "refund for outage", "k": 2},
                              headers={"x-api-key": "test-key"})
    assert r.status_code == 200
    body = r.json()
    assert len(body["hits"]) <= 2 and body["hits"][0]["section"] == "Refunds"


def test_validation(tmp_path):
    r = client(tmp_path).post("/query", json={"question": "x"}, headers={"x-api-key": "test-key"})
    assert r.status_code == 422
