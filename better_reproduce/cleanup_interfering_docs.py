"""Remove documents from an index that don't belong to it.

Usage:
    python cleanup_interfering_docs.py

Set INDEX_FOLDER and EXPECTED_DOC_CONTENT_CHARS to match the target index.
Run once per contaminated folder; safe to re-run (skips already-removed docs).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from lightrag import LightRAG
from lightrag.llm.ollama import ollama_model_complete
from lightrag.utils import EmbeddingFunc
from sentence_transformers import SentenceTransformer

# =========================
# Target index
# =========================
BASE_DIR = Path(__file__).resolve().parent
INDEX_FOLDER = BASE_DIR / "results" / "1_indexing" / "legal_gemma31b_no_reason_bad"

# Expected total character count for the one legitimate document in this index.
# Any document whose content_length differs is considered interfering.
EXPECTED_DATASET = "legal"
EXPECTED_CONTENT_CHARS = 21_466_292   # legal_unique_contexts joined length

# =========================
# Must match 1_indexing.py
# =========================
OLLAMA_HOST = "http://localhost:11440"
LLM = "gemma4:31b"
TIMEOUT_SECONDS = 1001
NUM_CTX = 8192
ENABLE_REASONING = False
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DEVICE = "cuda:1"
EMBEDDING_DIM = 1024
EMBED_BATCH_SIZE = 512
MAX_EMBED_TOKENS = 8192

MODEL = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE)


async def torch_embedding_func(texts: list[str]):
    return MODEL.encode(
        texts,
        normalize_embeddings=True,
        batch_size=EMBED_BATCH_SIZE,
        convert_to_numpy=True,
    )


async def main() -> None:
    doc_status_path = INDEX_FOLDER / "kv_store_doc_status.json"
    if not doc_status_path.exists():
        print(f"No doc_status store found at {doc_status_path}. Nothing to do.")
        return

    doc_status = json.loads(doc_status_path.read_text())
    intruders = [
        doc_id for doc_id, meta in doc_status.items()
        if meta.get("content_length") != EXPECTED_CONTENT_CHARS
    ]

    if not intruders:
        print("No interfering documents found. Index is clean.")
        return

    print(f"Found {len(intruders)} interfering document(s) to remove:")
    for doc_id in intruders:
        meta = doc_status[doc_id]
        print(f"  {doc_id}  content_length={meta.get('content_length')}  created={meta.get('created_at')}")

    rag = LightRAG(
        working_dir=str(INDEX_FOLDER),
        llm_model_func=ollama_model_complete,
        llm_model_name=LLM,
        llm_model_kwargs={
            "host": OLLAMA_HOST,
            "options": {"num_ctx": NUM_CTX},
            "timeout": TIMEOUT_SECONDS,
            "think": ENABLE_REASONING,
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=MAX_EMBED_TOKENS,
            func=torch_embedding_func,
        ),
    )
    await rag.initialize_storages()

    try:
        for doc_id in intruders:
            print(f"\nDeleting {doc_id} ...")
            result = await rag.adelete_by_doc_id(doc_id)
            print(f"  status={result.status}  message={result.message}")
    finally:
        await rag.finalize_storages()

    # Verify
    updated = json.loads(doc_status_path.read_text())
    remaining_intruders = [
        doc_id for doc_id in intruders if doc_id in updated
    ]
    if remaining_intruders:
        print(f"\nWARNING: {len(remaining_intruders)} doc(s) still present after deletion.")
    else:
        print(f"\nDone. Index now contains {len(updated)} document(s).")


if __name__ == "__main__":
    asyncio.run(main())
