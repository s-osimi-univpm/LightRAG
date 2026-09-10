import asyncio
import os
import inspect
import logging
import logging.config
from functools import partial
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
import json
from sentence_transformers import SentenceTransformer


#from dotenv import load_dotenv

#load_dotenv(dotenv_path=".env", override=False)

# === CONFIG ===
DATASET_NAME = ["agriculture"]#["mix"]#["agriculture","cs","legal","mix"]#"agriculture"
#INPUT_FILE = f"reproduce/dataset/unique_contexts/{DATASET_NAME}_unique_contexts.json"
#WORKING_DIR = f"reproduce/results/{DATASET_NAME}"
#LOG_FILE = f"{WORKING_DIR}/run.log"


def configure_logging(working_dir: str):
    """Configure logging for the application"""

    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    # Save logs inside the dataset working directory
    log_dir = working_dir
    log_file_path = os.path.abspath(os.path.join(log_dir, "step_1.log"))

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
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")


model = SentenceTransformer("BAAI/bge-m3", device="cuda:1")
async def torch_embedding_func(texts):
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,  
        batch_size=512,              
        convert_to_numpy=True
    )
    return embeddings

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name=os.getenv("LLM_MODEL", "gemma4:e4b-it-q4_K_M"),
        #summary_max_tokens=4096,
        llm_model_kwargs={
            "host": os.getenv("LLM_BINDING_HOST", "http://localhost:11440"),
            "options": {
                "num_ctx": 8192
                },
            "timeout": int(os.getenv("TIMEOUT", "1001"))
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


async def main(INPUT_FILE, WORKING_DIR):
    try:
        # Clear old data files
        files_to_delete = [
            "graph_chunk_entity_relation.graphml",
            "kv_store_doc_status.json",
            "kv_store_full_docs.json",
            "kv_store_text_chunks.json",
            "vdb_chunks.json",
            "vdb_entities.json",
            "vdb_relationships.json",
        ]

        """for file in files_to_delete:
            file_path = os.path.join(WORKING_DIR, file)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleting old file:: {file_path}")"""

        # Initialize RAG instance
        rag = await initialize_rag()

        # Test embedding function
        test_text = ["This is a test string for embedding."]
        embedding = await rag.embedding_func(test_text)
        embedding_dim = embedding.shape[1]
        print("\n=======================")
        print(f"PROCESSING {INPUT_FILE}")
        print("\n=======================")
        print("Test embedding function")
        print("========================")
        print(f"Test dict: {test_text}")
        print(f"Detected embedding dimension: {embedding_dim}\n\n")
       
        with open(INPUT_FILE, mode="r") as f:
            unique_contexts = json.load(f)

        # ✅ Converti lista -> testo continuo
        text = "\n\n".join(unique_contexts)
        await rag.ainsert(text)

        print("\n=====================")
        print("INSERTION OVER!! :)")
        print("=====================")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if rag:
            await rag.llm_response_cache.index_done_callback()
            await rag.finalize_storages()


if __name__ == "__main__":
    for cls in DATASET_NAME:
        INPUT_FILE = f"reproduce/dataset/unique_contexts/{cls}_unique_contexts.json"
        WORKING_DIR = f"reproduce/results/agriculture_legal_gemma4b"#{cls}"
        os.makedirs(WORKING_DIR, exist_ok=True)
        # Configure logging per dataset so logs are saved in the correct subfolder
        configure_logging(WORKING_DIR)
        asyncio.run(main(INPUT_FILE, WORKING_DIR))
    print("\nDone!")