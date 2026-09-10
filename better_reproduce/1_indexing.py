from __future__ import annotations

import asyncio
import json
from pathlib import Path

from lightrag import LightRAG
from lightrag.llm.ollama import ollama_model_complete
from lightrag.utils import EmbeddingFunc
from sentence_transformers import SentenceTransformer

try:
    from .functions import initialize_logger, write_constants_snapshot
except ImportError:
    from functions import initialize_logger, write_constants_snapshot


# =========================
# Experiment constants
# =========================
EXPERIMENT_NAME = "gemma31b_no_reason"

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset" / "unique_contexts"
RESULT_ROOT_DIR = BASE_DIR / "results" / "1_indexing"

DATASET_NAMES = ["legal"]

# LLM / host
OLLAMA_HOST = "http://localhost:11440"
LLM = "gemma4:31b"
TIMEOUT_SECONDS = 1001
NUM_CTX = 8192
# Gemma 4 models can emit reasoning by default in Ollama.
# Set to False to disable reasoning/thinking output.
ENABLE_REASONING = False

# Embedding
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DEVICE = "cuda:1"
EMBEDDING_DIM = 1024
EMBED_BATCH_SIZE = 512
MAX_EMBED_TOKENS = 8192

# Runtime behavior
VERBOSE_DEBUG = False
OVERWRITE_EXISTING_INDEX = True
CONTINUE_ON_DATASET_ERROR = True
# Chunking strategy for ingestion pipeline.
# Accepts selector chars: "F", "R", "V", "P"
# Or friendly names: "fixed_token", "recursive_character", "semantic_vector", "paragraph_semantic".
CHUNKING_STRATEGY = "F"
LIGHTRAG_STORE_FILES = [
    "graph_chunk_entity_relation.graphml",
    "kv_store_doc_status.json",
    "kv_store_full_docs.json",
    "kv_store_text_chunks.json",
    "kv_store_full_entities.json",
    "kv_store_full_relations.json",
    "kv_store_entity_chunks.json",
    "kv_store_relation_chunks.json",
    "kv_store_llm_response_cache.json",
    "vdb_chunks.json",
    "vdb_entities.json",
    "vdb_relationships.json",
]


MODEL = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE)


async def torch_embedding_func(texts: list[str]):
    return MODEL.encode(
        texts,
        normalize_embeddings=True,
        batch_size=EMBED_BATCH_SIZE,
        convert_to_numpy=True,
    )


def dataset_input_file(dataset_name: str) -> Path:
    return DATASET_DIR / f"{dataset_name}_unique_contexts.json"


def dataset_working_dir(dataset_name: str) -> Path:
    return RESULT_ROOT_DIR / f"{dataset_name}_{EXPERIMENT_NAME}"


def has_existing_rag_store(working_dir: Path) -> bool:
    for filename in LIGHTRAG_STORE_FILES:
        if (working_dir / filename).exists():
            return True
    return False


def clear_previous_rag_store(working_dir: Path) -> None:
    for filename in LIGHTRAG_STORE_FILES:
        target_file = working_dir / filename
        if target_file.exists():
            target_file.unlink()
            print(f"Deleted previous LightRAG file: {target_file}")


async def initialize_rag(working_dir: Path) -> LightRAG:
    rag = LightRAG(
        working_dir=str(working_dir),
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
    return rag


def resolve_chunking_strategy(value: str) -> str:
    aliases = {
        "fixed_token": "F",
        "recursive_character": "R",
        "semantic_vector": "V",
        "paragraph_semantic": "P",
    }
    normalized = value.strip().upper()
    if normalized in {"F", "R", "V", "P"}:
        return normalized

    normalized_name = value.strip().lower()
    if normalized_name in aliases:
        return aliases[normalized_name]

    raise ValueError(
        f"Unsupported CHUNKING_STRATEGY: {value!r}. "
        "Use one of F/R/V/P or fixed_token/recursive_character/semantic_vector/paragraph_semantic."
    )


async def run_dataset(dataset_name: str) -> None:
    input_file = dataset_input_file(dataset_name)
    working_dir = dataset_working_dir(dataset_name)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    working_dir.mkdir(parents=True, exist_ok=True)

    if has_existing_rag_store(working_dir):
        if OVERWRITE_EXISTING_INDEX:
            print(f"[{dataset_name}] existing index files found in {working_dir}; deleting before processing")
            clear_previous_rag_store(working_dir)
        else:
            raise RuntimeError(
                f"[{dataset_name}] existing index files found in {working_dir}; refusing to continue "
                "to avoid mixed/stale index content. Set OVERWRITE_EXISTING_INDEX=True or clear the folder first."
            )

    initialize_logger(working_dir, log_filename="step_1.log", verbose_debug=VERBOSE_DEBUG)
    chunking_strategy = resolve_chunking_strategy(CHUNKING_STRATEGY)

    rag = None
    try:
        rag = await initialize_rag(working_dir)

        with input_file.open("r", encoding="utf-8") as f:
            unique_contexts = json.load(f)

        text = "\n\n".join(unique_contexts)
        track_id = await rag.apipeline_enqueue_documents(
            text,
            process_options=chunking_strategy,
        )
        await rag.apipeline_process_enqueue_documents()

        print(
            f"[{dataset_name}] insertion completed with {len(unique_contexts)} contexts "
            f"(chunking={chunking_strategy}, track_id={track_id})"
        )

    finally:
        if rag is not None:
            await rag.llm_response_cache.index_done_callback()
            await rag.finalize_storages()


async def main_async() -> None:
    RESULT_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_name in DATASET_NAMES:
        dataset_dir = dataset_working_dir(dataset_name)
        constants_file = write_constants_snapshot(dataset_dir, globals())
        print(f"[{dataset_name}] constants snapshot written to: {constants_file}")

        try:
            await run_dataset(dataset_name)
        except Exception as exc:
            print(f"[{dataset_name}] failed: {exc}")
            if not CONTINUE_ON_DATASET_ERROR:
                raise


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
