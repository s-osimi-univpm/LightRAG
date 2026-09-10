from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError
from sentence_transformers import SentenceTransformer

from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete
from lightrag.utils import EmbeddingFunc

try:
    from .functions import initialize_logger, write_constants_snapshot
except ImportError:
    from functions import initialize_logger, write_constants_snapshot


# =========================
# Experiment constants
# =========================
EXPERIMENT_NAME = "rag_gemma31b_no_reason"

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
INDEXING_RESULTS_ROOT = RESULTS_DIR / "1_indexing"
QUESTIONS_RESULTS_ROOT = RESULTS_DIR / "2_questions"
RAG_RESULTS_ROOT = RESULTS_DIR / "3_rag"

DATASET_NAMES = ["agriculture"]

# Choose which step-1 indexing folder and step-2 questions folder to use for each dataset.
INDEXING_RESULT_FOLDER_BY_DATASET = {
    "agriculture": "agriculture_gemma31b_no_reason",
    #"legal": "legal_gemma31b_no_reason",
    #"mix": "mix_gemma31b_no_reason",
}
QUESTIONS_RESULT_FOLDER_BY_DATASET = {
    "agriculture": "agriculture_gemma31b_no_reason",
    #"legal": "legal_gemma31b_no_reason",
    #"mix": "mix_gemma31b_no_reason",
}


QUESTIONS_FILE_NAME = "questions.json"
OUTPUT_FILE_NAME = "rag_results.json"
OUTPUT_ERROR_FILE_NAME = "rag_errors.json"

# LLM / embedding
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11440")
LLM_MODEL = "gemma4:31b"
NUM_CTX = 40000
REQUEST_TIMEOUT_SECONDS = 2400
SENTENCE_TRANSFORMER_MODEL = "BAAI/bge-m3"
SENTENCE_TRANSFORMER_DEVICE = "cuda:1"

# Query behavior
QUERY_MODES = ["hybrid", "naive", "mix"]
ENABLE_RERANK = False

# Runtime behavior
VERBOSE_DEBUG = False


class GeneratedQuestion(BaseModel):
    question: str = Field(..., min_length=1)
    reference_answer: str = Field(..., min_length=1)
    top_supporting_chunk_ids: list[str] = Field(default_factory=list)


class TaskQuestions(BaseModel):
    task_description: str = Field(..., min_length=1)
    generated_questions: list[GeneratedQuestion] = Field(default_factory=list)


class UserQuestions(BaseModel):
    user_description: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    sampled_chunks: list[dict[str, Any]] = Field(default_factory=list)
    task_questions: list[TaskQuestions] = Field(default_factory=list)


class DatasetQuestionsOutput(BaseModel):
    dataset_name: str = Field(..., min_length=1)
    users: list[UserQuestions] = Field(default_factory=list)


class QueryWorkItem(BaseModel):
    dataset_name: str
    user_description: str
    user_query: str
    task_description: str
    question: str
    reference_answer: str
    fundamental_chunk_ids: list[str] = Field(default_factory=list)


class StructuredRagAnswer(BaseModel):
    answer: str = Field(..., min_length=1)
    confidence: float | None = Field(default=None)
    assumptions: list[str] = Field(default_factory=list)


class RagResultEntry(BaseModel):
    mode: str
    dataset_name: str
    user_description: str
    user_query: str
    task_description: str
    question: str
    reference_answer: str
    fundamental_chunk_count: int
    retrieved_chunk_count: int
    graph_hit_chunks: int
    chunk_hit_count: int
    graph_hit_count: int
    retrieved_context_tokens: int
    retrieved_context: dict[str, Any]
    structured_answer: StructuredRagAnswer
    raw_result: str


class RagErrorEntry(BaseModel):
    mode: str
    dataset_name: str
    user_description: str
    task_description: str
    question: str
    error: str


class RagRunOutput(BaseModel):
    dataset_name: str
    indexing_folder: str
    questions_folder: str
    total_questions: int
    results: list[RagResultEntry]
    errors: list[RagErrorEntry]


_embedding_model = SentenceTransformer(
    SENTENCE_TRANSFORMER_MODEL,
    device=SENTENCE_TRANSFORMER_DEVICE,
)


async def torch_embedding_func(texts: list[str]):
    embeddings = _embedding_model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=128,
        convert_to_numpy=True,
    )
    return embeddings


def dataset_output_dir(dataset_name: str) -> Path:
    return RAG_RESULTS_ROOT / f"{dataset_name}_{EXPERIMENT_NAME}"


def questions_file_path(dataset_name: str) -> Path:
    folder = QUESTIONS_RESULT_FOLDER_BY_DATASET[dataset_name]
    return QUESTIONS_RESULTS_ROOT / folder / QUESTIONS_FILE_NAME


def indexing_working_dir(dataset_name: str) -> Path:
    folder = INDEXING_RESULT_FOLDER_BY_DATASET[dataset_name]
    return INDEXING_RESULTS_ROOT / folder


def load_structured_questions(dataset_name: str) -> list[QueryWorkItem]:
    file_path = questions_file_path(dataset_name)
    if not file_path.exists():
        raise FileNotFoundError(f"Questions file not found: {file_path}")

    payload = file_path.read_text(encoding="utf-8")
    parsed = DatasetQuestionsOutput.model_validate_json(payload)

    work_items: list[QueryWorkItem] = []
    for user in parsed.users:
        for task in user.task_questions:
            for generated in task.generated_questions:
                work_items.append(
                    QueryWorkItem(
                        dataset_name=parsed.dataset_name,
                        user_description=user.user_description,
                        user_query=user.query,
                        task_description=task.task_description,
                        question=generated.question,
                        reference_answer=generated.reference_answer,
                        fundamental_chunk_ids=generated.top_supporting_chunk_ids,
                    )
                )

    return work_items


def count_retrieved_context_tokens(raw_data: dict[str, Any], rag_instance: LightRAG) -> int:
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

    clean_chunks = [
        {"chunk_id": c.get("chunk_id", ""), "content": c.get("content", "")}
        for c in (chunks if isinstance(chunks, list) else [])
        if isinstance(c, dict)
    ]
    return {
        "entities": entities if isinstance(entities, list) else [],
        "relationships": relationships if isinstance(relationships, list) else [],
        "chunks": clean_chunks,
    }


def extract_chunk_ids_from_source_id(source_id: Any) -> set[str]:
    if not isinstance(source_id, str):
        return set()

    chunk_ids: set[str] = set()
    for part in source_id.split("<SEP>"):
        chunk_id = part.strip()
        if chunk_id:
            chunk_ids.add(chunk_id)
    return chunk_ids


def extract_retrieved_chunk_ids(
    retrieved_context: dict[str, Any],
    include_graph_links: bool = False,
) -> set[str]:
    chunks = retrieved_context.get("chunks", [])
    if not isinstance(chunks, list):
        chunks = []

    chunk_ids: set[str] = set()
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        if chunk_id:
            chunk_ids.add(chunk_id)

    if include_graph_links:
        entities = retrieved_context.get("entities", [])
        relationships = retrieved_context.get("relationships", [])

        if isinstance(entities, list):
            for entity in entities:
                if not isinstance(entity, dict):
                    continue
                chunk_ids.update(extract_chunk_ids_from_source_id(entity.get("source_id")))

        if isinstance(relationships, list):
            for rel in relationships:
                if not isinstance(rel, dict):
                    continue
                chunk_ids.update(extract_chunk_ids_from_source_id(rel.get("source_id")))

    return chunk_ids


def extract_graph_linked_chunk_ids(retrieved_context: dict[str, Any]) -> set[str]:
    chunk_ids: set[str] = set()

    entities = retrieved_context.get("entities", [])
    relationships = retrieved_context.get("relationships", [])

    if isinstance(entities, list):
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            chunk_ids.update(extract_chunk_ids_from_source_id(entity.get("source_id")))

    if isinstance(relationships, list):
        for rel in relationships:
            if not isinstance(rel, dict):
                continue
            chunk_ids.update(extract_chunk_ids_from_source_id(rel.get("source_id")))

    return chunk_ids


def compute_chunk_hit_metrics(
    fundamental_chunk_ids: list[str],
    retrieved_context: dict[str, Any],
) -> tuple[int, int, int, int, int]:
    fundamental_set = {cid.strip() for cid in fundamental_chunk_ids if cid.strip()}
    direct_set = extract_retrieved_chunk_ids(retrieved_context, include_graph_links=False)
    graph_set = extract_graph_linked_chunk_ids(retrieved_context)

    return (
        len(fundamental_set),
        len(direct_set),
        len(graph_set),
        len(fundamental_set.intersection(direct_set)),
        len(fundamental_set.intersection(graph_set)),
    )


def find_json_object(raw_text: str) -> dict[str, Any]:
    raw_text = raw_text.strip()
    if raw_text.startswith("{") and raw_text.endswith("}"):
        return json.loads(raw_text)

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM response")

    return json.loads(raw_text[start : end + 1])


def parse_structured_answer(raw_result: str) -> StructuredRagAnswer:
    try:
        payload = find_json_object(raw_result)
        parsed = StructuredRagAnswer.model_validate(payload)
        if parsed.confidence is not None:
            parsed.confidence = max(0.0, min(1.0, parsed.confidence))
        return parsed
    except (ValueError, json.JSONDecodeError, ValidationError):
        # Keep run robust if model occasionally emits non-JSON text.
        return StructuredRagAnswer(answer=raw_result.strip() or "No answer returned")


def append_json_array_entry(file_path: Path, entry: dict[str, Any]) -> None:
    """Append one entry to a JSON array file while keeping valid JSON format."""
    if not file_path.exists():
        file_path.write_text("[]", encoding="utf-8")

    current = file_path.read_text(encoding="utf-8").strip()
    if not current:
        current = "[]"

    entry_json = json.dumps(entry, ensure_ascii=False, indent=2)

    if current == "[]":
        file_path.write_text(f"[\n{entry_json}\n]", encoding="utf-8")
        return

    if not current.endswith("]"):
        raise ValueError(f"Invalid JSON array file: {file_path}")

    updated = f"{current[:-1]},\n{entry_json}\n]"
    file_path.write_text(updated, encoding="utf-8")


def build_structured_query_prompt(question: str) -> str:
    return (
        "Answer the question using the retrieved context. "
        "Return only JSON with this schema: "
        '{"answer": string, "confidence": number between 0 and 1, "assumptions": string[]}.'
        f"\nQuestion: {question}"
    )


async def initialize_rag(working_dir: Path) -> LightRAG:
    # Keep first-call startup manageable on a private single-GPU Ollama daemon.
    # You can override both values via environment when needed.
    runtime_num_ctx = int(os.getenv("RAG_NUM_CTX", str(NUM_CTX)))
    runtime_timeout_seconds = int(os.getenv("RAG_REQUEST_TIMEOUT_SECONDS", str(REQUEST_TIMEOUT_SECONDS)))

    rag = LightRAG(
        working_dir=str(working_dir),
        llm_model_func=ollama_model_complete,
        llm_model_name=LLM_MODEL,
        enable_llm_cache=False,
        llm_model_kwargs={
            "host": OLLAMA_HOST,
            "options": {"num_ctx": runtime_num_ctx},
            "timeout": runtime_timeout_seconds,
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
            max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8192")),
            func=torch_embedding_func,
        ),
    )

    await rag.initialize_storages()
    return rag


async def run_queries_for_mode(
    rag: LightRAG,
    mode: str,
    work_items: list[QueryWorkItem],
    result_file: Path,
    error_file: Path,
) -> tuple[list[RagResultEntry], list[RagErrorEntry]]:
    query_param = QueryParam(mode=mode, enable_rerank=ENABLE_RERANK)
    results: list[RagResultEntry] = []
    errors: list[RagErrorEntry] = []

    for index, item in enumerate(work_items, start=1):
        print(f"[{mode}] Query {index}/{len(work_items)}")

        try:
            query_response = await rag.aquery_llm(
                build_structured_query_prompt(item.question),
                param=query_param,
            )
            llm_response = query_response.get("llm_response", {})
            raw_result = str(llm_response.get("content", ""))

            parsed_answer = parse_structured_answer(raw_result)
            context_tokens = count_retrieved_context_tokens(query_response, rag)
            retrieved_context = extract_retrieved_context(query_response)
            (
                fundamental_chunk_count,
                retrieved_chunk_count,
                graph_hit_chunks,
                chunk_hit_count,
                graph_hit_count,
            ) = compute_chunk_hit_metrics(item.fundamental_chunk_ids, retrieved_context)

            results.append(
                RagResultEntry(
                    mode=mode,
                    dataset_name=item.dataset_name,
                    user_description=item.user_description,
                    user_query=item.user_query,
                    task_description=item.task_description,
                    question=item.question,
                    reference_answer=item.reference_answer,
                    fundamental_chunk_count=fundamental_chunk_count,
                    retrieved_chunk_count=retrieved_chunk_count,
                    graph_hit_chunks=graph_hit_chunks,
                    chunk_hit_count=chunk_hit_count,
                    graph_hit_count=graph_hit_count,
                    retrieved_context_tokens=context_tokens,
                    retrieved_context=retrieved_context,
                    structured_answer=parsed_answer,
                    raw_result=raw_result,
                )
            )
            append_json_array_entry(result_file, results[-1].model_dump())

        except Exception as exc:  # noqa: BLE001
            errors.append(
                RagErrorEntry(
                    mode=mode,
                    dataset_name=item.dataset_name,
                    user_description=item.user_description,
                    task_description=item.task_description,
                    question=item.question,
                    error=str(exc),
                )
            )
            append_json_array_entry(error_file, errors[-1].model_dump())

    return results, errors


async def run_dataset(dataset_name: str) -> None:
    output_dir = dataset_output_dir(dataset_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    initialize_logger(output_dir, log_filename="3_rag.log", verbose_debug=VERBOSE_DEBUG)
    runtime_num_ctx = int(os.getenv("RAG_NUM_CTX", str(NUM_CTX)))
    runtime_timeout_seconds = int(os.getenv("RAG_REQUEST_TIMEOUT_SECONDS", str(REQUEST_TIMEOUT_SECONDS)))
    print(
        f"[{dataset_name}] using OLLAMA_HOST={OLLAMA_HOST} model={LLM_MODEL} "
        f"num_ctx={runtime_num_ctx} timeout_s={runtime_timeout_seconds}"
    )

    indexing_dir = indexing_working_dir(dataset_name)
    if not indexing_dir.exists():
        raise FileNotFoundError(f"Indexing folder not found: {indexing_dir}")

    work_items = load_structured_questions(dataset_name)
    if not work_items:
        raise ValueError(f"No questions found in {questions_file_path(dataset_name)}")

    rag = await initialize_rag(indexing_dir)

    all_results: list[RagResultEntry] = []
    all_errors: list[RagErrorEntry] = []

    output_file = output_dir / OUTPUT_FILE_NAME
    error_file = output_dir / OUTPUT_ERROR_FILE_NAME
    # Initialize files as JSON arrays so each step can append valid entries.
    output_file.write_text("[]", encoding="utf-8")
    error_file.write_text("[]", encoding="utf-8")

    for mode in QUERY_MODES:
        mode_results, mode_errors = await run_queries_for_mode(
            rag,
            mode,
            work_items,
            output_file,
            error_file,
        )
        all_results.extend(mode_results)
        all_errors.extend(mode_errors)

    run_output = RagRunOutput(
        dataset_name=dataset_name,
        indexing_folder=INDEXING_RESULT_FOLDER_BY_DATASET[dataset_name],
        questions_folder=QUESTIONS_RESULT_FOLDER_BY_DATASET[dataset_name],
        total_questions=len(work_items),
        results=all_results,
        errors=all_errors,
    )

    summary_file = output_dir / "rag_run_summary.json"
    summary_file.write_text(run_output.model_dump_json(indent=2), encoding="utf-8")

    print(f"[{dataset_name}] structured RAG results written to {output_file}")
    print(f"[{dataset_name}] errors written to {error_file}")
    print(f"[{dataset_name}] run summary written to {summary_file}")


async def run_all_datasets() -> None:
    RAG_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    for dataset_name in DATASET_NAMES:
        if dataset_name not in INDEXING_RESULT_FOLDER_BY_DATASET:
            raise KeyError(f"Missing indexing folder mapping for dataset: {dataset_name}")
        if dataset_name not in QUESTIONS_RESULT_FOLDER_BY_DATASET:
            raise KeyError(f"Missing questions folder mapping for dataset: {dataset_name}")

        output_dir = dataset_output_dir(dataset_name)
        constants_file = write_constants_snapshot(output_dir, globals())
        print(f"[{dataset_name}] constants snapshot written to: {constants_file}")
        await run_dataset(dataset_name)

    print("Done!")


def main() -> None:
    # Keep one event loop for the full run to avoid cross-loop lock reuse inside LightRAG workers.
    asyncio.run(run_all_datasets())


if __name__ == "__main__":
    main()
