import pytest

from rag.chunking import chunk_document
from rag.ingest import Document, content_id, normalize


def make(text):
    return Document(content_id(text), "x.md", text)


def test_normalize_collapses_blank_lines_and_controls():
    assert normalize("a\x00b\n\n\n\nc  \n") == "ab\n\nc"


def test_chunks_respect_size_and_sections():
    body = " ".join(f"sentence {i}." for i in range(400))
    doc = make(f"# Title\n\n## Alpha\n\n{body}\n\n## Beta\n\nshort text")
    chunks = chunk_document(doc, size=300, overlap=50)
    assert all(len(c.text) <= 300 + 1 for c in chunks)
    assert {c.section for c in chunks} == {"Alpha", "Beta"}
    assert chunks[-1].text == "short text"


def test_overlap_carries_context_between_chunks():
    doc = make(" ".join(f"w{i}" for i in range(300)))
    a, b = chunk_document(doc, size=200, overlap=40)[:2]
    assert a.text[-20:] in b.text


def test_chunk_ids_are_stable():
    doc = make("## S\n\nhello world")
    assert chunk_document(doc) == chunk_document(doc)


def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_document(make("x"), size=100, overlap=100)
