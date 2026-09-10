import re
import json
import os
import asyncio
import logging
import logging.config
from typing import Any
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc, always_get_an_event_loop
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
import json
from sentence_transformers import SentenceTransformer

# === CONFIG ===
DATASET_NAME = ["agriculture"]#["mix"]#["agriculture","legal"]
#INPUT_FILE = f"reproduce/dataset/unique_contexts/{DATASET_NAME}_unique_contexts.json"
#LOG_DIR = f"reproduce/results"
#LOG_FILE = f"{WORKING_DIR}/run.log"


def configure_logging(WORKING_DIR):
    """Configure logging for the application"""

    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

        # Save logs inside the dataset working directory
    log_dir = WORKING_DIR

    os.makedirs(log_dir, exist_ok=True)

    log_file_path = os.path.join(
        log_dir,
        "lightrag_step3.log"
    )

    print(f"\nLightRAG compatible demo log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(True)


#if not os.path.exists(WORKING_DIR):
#    os.mkdir(WORKING_DIR)

model = SentenceTransformer("BAAI/bge-m3", device="cuda:1")
async def torch_embedding_func(texts):
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,  
        batch_size=128,              
        convert_to_numpy=True
    )
    return embeddings

async def initialize_rag(working_dir):
    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=ollama_model_complete,
        llm_model_name=os.getenv("LLM_MODEL", "gemma4:31b"), #"gemma4:12b"
        #summary_max_tokens=8192,
        enable_llm_cache=False,
        llm_model_kwargs={
            "host": os.getenv("LLM_BINDING_HOST", "http://localhost:11440"),
            "options": {"num_ctx": 40000}, #32000*4 #130000
            "timeout": int(os.getenv("TIMEOUT", "1001")),
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
            max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8192")),
            func=torch_embedding_func,
        ),
    )

    await rag.initialize_storages()  # Auto-initializes pipeline_status
    return rag


async def print_stream(stream):
    async for chunk in stream:
        print(chunk, end="", flush=True)


def extract_queries(file_path):
    with open(file_path, "r") as f:
        data = f.read()

    data = data.replace("**", "")

    queries = re.findall(r"\*\s*Question\s+\d+:\s*(.+)", data)

    return queries


def count_retrieved_context_tokens(raw_data: dict[str, Any], rag_instance: LightRAG) -> int:
    """Count tokens from retrieved context payload returned by aquery_llm."""
    tokenizer = getattr(rag_instance, "tokenizer", None)
    if tokenizer is None:
        return 0

    data_section = raw_data.get("data", {}) if isinstance(raw_data, dict) else {}

    entities = data_section.get("entities", [])
    relationships = data_section.get("relationships", [])
    chunks = data_section.get("chunks", [])

    context_parts: list[str] = []

    for entity in entities:
        if not isinstance(entity, dict):
            continue
        context_parts.append(
            " | ".join(
                [
                    str(entity.get("entity_name", "")),
                    str(entity.get("description", "")),
                ]
            )
        )

    for rel in relationships:
        if not isinstance(rel, dict):
            continue
        context_parts.append(
            " | ".join(
                [
                    str(rel.get("src_id", "")),
                    str(rel.get("tgt_id", "")),
                    str(rel.get("description", "")),
                    str(rel.get("keywords", "")),
                ]
            )
        )

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        context_parts.append(str(chunk.get("content", "")))

    context_text = "\n".join(part for part in context_parts if part)
    if not context_text:
        return 0

    return len(tokenizer.encode(context_text))


def extract_retrieved_context(raw_data: dict[str, Any]) -> dict[str, Any]:
    """Extract retrieved context payload from aquery_llm raw response."""
    if not isinstance(raw_data, dict):
        return {
            "entities": [],
            "relationships": [],
            "chunks": [],
        }

    data_section = raw_data.get("data", {})
    if not isinstance(data_section, dict):
        data_section = {}

    entities = data_section.get("entities", [])
    relationships = data_section.get("relationships", [])
    chunks = data_section.get("chunks", [])

    return {
        "entities": entities if isinstance(entities, list) else [],
        "relationships": relationships if isinstance(relationships, list) else [],
        "chunks": chunks if isinstance(chunks, list) else [],
    }

async def run_queries_and_save_to_json(
    queries, rag_instance, query_param, output_file, error_file
):

    logger.info(f"Starting mode: {query_param.mode}")

    with (
        open(output_file, "w", encoding="utf-8") as result_file,
        open(error_file, "a", encoding="utf-8") as err_file,
    ):

        result_file.write("[\n")
        first_entry = True

        for idx, query_text in enumerate(queries):

            logger.info(f"[{query_param.mode}] Query {idx+1}/{len(queries)}")
            logger.info(f"Question: {query_text}")

            try:

                query_response = await rag_instance.aquery_llm(
                    query_text,
                    param=query_param
                )
                llm_response = query_response.get("llm_response", {})
                result = llm_response.get("content", "")
                context_tokens = count_retrieved_context_tokens(query_response, rag_instance)
                retrieved_context = extract_retrieved_context(query_response)

                logger.info(
                    f"[{query_param.mode}] Retrieved context tokens: {context_tokens}"
                )

                print("=" * 80)
                print("MODE:", query_param.mode)
                print("QUERY:", query_text)
                print("RETRIEVED CONTEXT TOKENS:", context_tokens)
                print("RESULT TYPE:", type(result))
                print("RESULT:")
                print(repr(result))
                print("=" * 80)

                if result is None:
                    logger.warning(
                        f"[{query_param.mode}] NULL RESULT "
                        f"for query: {query_text}"
                    )

                elif isinstance(result, str) and result.strip() == "":
                    logger.warning(
                        f"[{query_param.mode}] EMPTY STRING RESULT "
                        f"for query: {query_text}"
                    )

                else:
                    logger.info(
                        f"[{query_param.mode}] Successful response"
                    )

                if not first_entry:
                    result_file.write(",\n")

                json.dump(
                    {
                        "query": query_text,
                        "retrieved_context_tokens": context_tokens,
                        "retrieved_context": retrieved_context,
                        "result": result,
                    },
                    result_file,
                    ensure_ascii=False,
                    indent=4,
                )

                first_entry = False

            except Exception as e:

                logger.exception(
                    f"[{query_param.mode}] ERROR "
                    f"for query: {query_text}"
                )

                json.dump(
                    {
                        "query": query_text,
                        "error": str(e),
                    },
                    err_file,
                    ensure_ascii=False,
                    indent=4,
                )

                err_file.write("\n")

        result_file.write("\n]")

    logger.info(f"Completed mode: {query_param.mode}")

async def main(cls):
    configure_logging(f"reproduce/results/{cls}")

    rag = await initialize_rag(f"reproduce/results/agriculture_legal_gemma4b")#{cls}")

    queries = extract_queries(
        f"reproduce/questions/{cls}_questions.txt"
    )

    os.makedirs("reproduce/results", exist_ok=True)

    print("Running hybrid mode...")
    await run_queries_and_save_to_json(
        queries,
        rag,
        QueryParam(
            mode="hybrid",
            enable_rerank=False,
            #only_need_prompt=True,
            #top_k=5
            ),
        f"reproduce/results/{cls}agr_hybrid_results.json",
        f"reproduce/results/{cls}agr_hybrid_results_errors.json"
    )

    print("Running naive mode...")
    await run_queries_and_save_to_json(
        queries,
        rag,
        QueryParam(
            mode="naive",
            enable_rerank=False
            ),
        f"reproduce/results/{cls}agr_naive_results.json",
        f"reproduce/results/{cls}agr_naive_results_errors.json"
    )

    
if __name__ == "__main__":
    for cls in DATASET_NAME:
        INPUT_FILE = f"reproduce/dataset/unique_contexts/{cls}_unique_contexts.json"
        WORKING_DIR = f"reproduce/results/{cls}"
        #configure_logging(WORKING_DIR)
        if not os.path.exists(WORKING_DIR):
            os.mkdir(WORKING_DIR)
        asyncio.run(main(cls))
    print("\nDone!")