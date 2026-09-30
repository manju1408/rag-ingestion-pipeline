from generate_docs import main as generate
from rag.pipeline import build_index, build_prompt
from rag.store import VectorStore
from rag.embeddings import HashingEmbedder


def test_retrieves_relevant_section(tmp_path):
    generate(str(tmp_path))
    store = build_index(tmp_path)
    (top, score), *_ = store.search("how many days do I have to return my modem", k=3)
    assert top.section == "Return window"
    assert score > 0


def test_reingest_is_idempotent(tmp_path):
    generate(str(tmp_path))
    store = build_index(tmp_path)
    n = len(store)
    build_index(tmp_path, store=store)
    assert len(store) == n


def test_persistence_roundtrip(tmp_path):
    generate(str(tmp_path / "docs"))
    store = build_index(tmp_path / "docs")
    store.save(tmp_path / "index")
    loaded = VectorStore.load(tmp_path / "index", HashingEmbedder())
    q = "autopay discount"
    assert [c.chunk_id for c, _ in loaded.search(q)] == [c.chunk_id for c, _ in store.search(q)]


def test_prompt_is_grounded(tmp_path):
    generate(str(tmp_path))
    hits = build_index(tmp_path).search("late fee", k=2)
    prompt = build_prompt("What is the late fee?", hits)
    assert "[1]" in prompt and "only the context" in prompt
