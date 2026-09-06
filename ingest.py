from pathlib import Path

from agno.knowledge.knowledge import Knowledge
from agno.vectordb.qdrant import Qdrant
from agno.knowledge.embedder.ollama import OllamaEmbedder


# ============================================================
# CONFIG
# ============================================================

QDRANT_URL = "http://localhost:6333"

COLLECTION_NAME = "bioretrieve-healthcare-multidoc"

EMBEDDING_MODEL = "nomic-embed-text"

DOCS_DIR = Path("docs")


# ============================================================
# QDRANT
# ============================================================

vector_db = Qdrant(
    collection=COLLECTION_NAME,
    url=QDRANT_URL,
    embedder=OllamaEmbedder(
        id=EMBEDDING_MODEL,
        dimensions=768,
    ),
)


# ============================================================
# KNOWLEDGE BASE
# ============================================================

knowledge_base = Knowledge(
    vector_db=vector_db,
)


# ============================================================
# FIND PDFs
# ============================================================

pdf_files = [
    pdf
    for pdf in sorted(DOCS_DIR.glob("*.pdf"))
    if pdf.stat().st_size > 0
]


if not pdf_files:

    raise RuntimeError(
        "No valid PDF documents found in docs/"
    )


# ============================================================
# INDEX DOCUMENTS
# ============================================================

for pdf in pdf_files:

    print(f"Indexing: {pdf.name}")

    knowledge_base.insert(
        path=str(pdf),
        skip_if_exists=True,
    )


print()
print("==========================================")
print("✅ Knowledge base indexing complete")
print("==========================================")