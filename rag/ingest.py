"""Load source documents and give each a stable, content-derived id."""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED = {".md", ".txt"}


@dataclass(frozen=True)
class Document:
    doc_id: str
    source: str
    text: str
    metadata: dict = field(default_factory=dict)


def normalize(text: str) -> str:
    """Unicode-normalize, strip control chars and collapse runs of blank lines."""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def content_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_documents(root: str | Path) -> list[Document]:
    """Read every supported file under ``root``. Identical content -> identical id,
    so re-ingesting the same folder is idempotent."""
    docs: dict[str, Document] = {}
    for path in sorted(Path(root).rglob("*")):
        if path.suffix.lower() not in SUPPORTED or not path.is_file():
            continue
        text = normalize(path.read_text(encoding="utf-8", errors="replace"))
        if not text:
            continue
        title = next((l.lstrip("# ").strip() for l in text.splitlines() if l.startswith("#")), path.stem)
        doc = Document(content_id(text), str(path.relative_to(root)), text, {"title": title})
        docs.setdefault(doc.doc_id, doc)
    return list(docs.values())
