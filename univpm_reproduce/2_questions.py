from __future__ import annotations

import json
import os
import random
from pathlib import Path

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from instructor.v2.core.errors import IncompleteOutputException
from lightrag.utils import logger

try:
    from .functions import initialize_logger, write_constants_snapshot
except ImportError:
    from functions import initialize_logger, write_constants_snapshot


# =========================
# Experiment constants
# =========================
EXPERIMENT_NAME = "gemma31b_no_reason"

BASE_DIR = Path(__file__).resolve().parent
STEP_1_RESULTS_DIR = BASE_DIR / "results" / "1_indexing"
RESULT_ROOT_DIR = BASE_DIR / "results" / "2_questions"

DATASET_NAMES = ["agriculture","legal","mix"]

# LLM / host
OLLAMA_HOST = "http://localhost:11440"
LLM = "gemma4:31b"
NUM_CTX = 32768 
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "12288"))
REQUEST_TIMEOUT_SECONDS = 1001
OLLAMA_API_KEY = "ollama"
LLM_TEMPERATURE = 0.7

# Structured generation sizes
NUM_USERS = 5
TASKS_PER_USER = 5
QUESTIONS_PER_TASK = 5
MAX_GENERATION_RETRIES = 4

# Chunk sampling
STEP_1_EXPERIMENT_NAME = "gemma31b_no_reason"
RANDOM_SEED = 42
CHUNKS_PER_DATASET = 15
MIN_CHUNK_CHARS = 200
USER_CHUNK_POOL_FRACTION = 1 / NUM_USERS

# Runtime behavior
VERBOSE_DEBUG = False
OUTPUT_FILE_NAME = "questions.json"
OUTPUT_METADATA_FILE_NAME = "chunk_samples.json"

SIMPLE_USER_TASKS_QUESTIONS_PROMPT = """
Given the following dataset description:

{total_description}

Create exactly 1 user persona who would engage with this dataset.
For that user, create exactly {tasks_per_user} realistic tasks.
For each task, create exactly {questions_per_task} high-level, non-trivial questions that
require understanding the provided dataset description.

Important style constraints for each question:
- Write each question as if asked by a naive end user who has NOT seen the source chunks.
- Do NOT mention chunks, chunk IDs, sections, documents, file paths, or "the provided context".
- Do NOT cite or quote source text directly.
- Questions should sound natural and practical for the persona/task.

For every generated question, also provide top_supporting_chunk_ids:
- Include 1 to 3 chunk IDs that most strongly support why this question is relevant.
- Rank them from most relevant to less relevant.
- Use only IDs from this allowed set:
{allowed_chunk_ids}

For every generated question, also provide reference_answer:
- Write a concise, high-quality answer grounded in the dataset description.
- This will be used as ground-truth style reference text for RAG evaluation.
- Do NOT mention chunks, chunk IDs, sections, documents, file paths, or "the provided context".
""".strip()


class GeneratedQuestion(BaseModel):
    question: str = Field(..., min_length=1)
    reference_answer: str = Field(..., min_length=1)
    top_supporting_chunk_ids: list[str] = Field(..., min_length=1)


class TaskQuestions(BaseModel):
    task_description: str = Field(..., min_length=1)
    generated_questions: list[GeneratedQuestion] = Field(..., min_length=1)


class UserQuestionsResponse(BaseModel):
    user_description: str = Field(..., min_length=1)
    task_questions: list[TaskQuestions] = Field(..., min_length=1)


class UserOutput(BaseModel):
    user_description: str
    query: str
    sampled_chunks: list[dict[str, object]]
    task_questions: list[TaskQuestions]


class DatasetQuestionsOutput(BaseModel):
    dataset_name: str
    users: list[UserOutput]


def chunk_store_file(dataset_name: str) -> Path:
    return STEP_1_RESULTS_DIR / f"{dataset_name}_{STEP_1_EXPERIMENT_NAME}" / "kv_store_text_chunks.json"


def dataset_output_dir(dataset_name: str) -> Path:
    return RESULT_ROOT_DIR / f"{dataset_name}_{EXPERIMENT_NAME}"


def load_chunks(dataset_name: str) -> list[dict[str, object]]:
    chunk_file = chunk_store_file(dataset_name)
    if not chunk_file.exists():
        raise FileNotFoundError(f"Chunk store not found: {chunk_file}")

    with chunk_file.open("r", encoding="utf-8") as f:
        chunk_store = json.load(f)

    chunks: list[dict[str, object]] = []
    for chunk_id, chunk_data in chunk_store.items():
        content = chunk_data.get("content")
        if not isinstance(content, str) or len(content.strip()) < MIN_CHUNK_CHARS:
            continue

        chunks.append(
            {
                "chunk_id": chunk_id,
                "chunk_order_index": chunk_data.get("chunk_order_index"),
                "full_doc_id": chunk_data.get("full_doc_id"),
                "file_path": chunk_data.get("file_path"),
                "tokens": chunk_data.get("tokens"),
                "content": content,
            }
        )

    if not chunks:
        raise ValueError(f"No usable chunks found in: {chunk_file}")

    return chunks


def sort_chunks_by_order(chunks: list[dict[str, object]]) -> list[dict[str, object]]:
    def sort_key(chunk: dict[str, object]) -> tuple[int, str, str]:
        raw_order_index = chunk.get("chunk_order_index")
        if isinstance(raw_order_index, int):
            order_index = raw_order_index
        elif isinstance(raw_order_index, str):
            try:
                order_index = int(raw_order_index)
            except ValueError:
                order_index = 10**12
        else:
            order_index = 10**12

        full_doc_id = str(chunk.get("full_doc_id") or "")
        chunk_id = str(chunk.get("chunk_id") or "")
        return (order_index, full_doc_id, chunk_id)

    return sorted(chunks, key=sort_key)


def build_user_chunk_pools(dataset_name: str) -> list[list[dict[str, object]]]:
    ordered_chunks = sort_chunks_by_order(load_chunks(dataset_name))

    total_chunks = len(ordered_chunks)
    base_pool_size = total_chunks // NUM_USERS
    remainder = total_chunks % NUM_USERS

    user_pools: list[list[dict[str, object]]] = []
    start = 0
    for user_idx in range(NUM_USERS):
        pool_size = base_pool_size + (1 if user_idx < remainder else 0)
        end = start + pool_size
        user_pools.append(ordered_chunks[start:end])
        start = end

    return user_pools


def sample_chunks_for_user(
    user_chunks: list[dict[str, object]],
    dataset_name: str,
    user_index: int,
) -> list[dict[str, object]]:
    if not user_chunks:
        raise ValueError(
            f"No chunks available for user {user_index} in dataset '{dataset_name}'"
        )

    ordered_user_chunks = sort_chunks_by_order(user_chunks)
    sample_size = min(CHUNKS_PER_DATASET, len(ordered_user_chunks))
    if sample_size == len(ordered_user_chunks):
        return ordered_user_chunks

    max_start = len(ordered_user_chunks) - sample_size
    rng = random.Random(f"{RANDOM_SEED}:{dataset_name}:user:{user_index}")
    start = rng.randint(0, max_start)
    end = start + sample_size
    return ordered_user_chunks[start:end]


def build_description_from_chunks(sampled_chunks: list[dict[str, object]]) -> str:
    parts: list[str] = []
    for idx, chunk in enumerate(sampled_chunks, start=1):
        parts.append(
            (
                f"[Chunk {idx}] id={chunk['chunk_id']} "
                f"order={chunk.get('chunk_order_index')} "
                f"doc={chunk.get('full_doc_id')}\n"
                f"{chunk['content']}"
            )
        )
    return "\n\n".join(parts)


def _normalize_chunk_id(raw_id: object) -> str:
    text = str(raw_id).strip().strip('"').strip("'")
    text = text.replace("`", "")
    return text


def _build_allowed_chunk_id_lookup(allowed_chunk_ids: list[str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for chunk_id in allowed_chunk_ids:
        canonical = str(chunk_id)
        normalized = _normalize_chunk_id(canonical)
        lowered = normalized.lower()
        lookup[normalized] = canonical
        lookup[lowered] = canonical

        # Common LLM shortening patterns: only the trailing chunk index token.
        last_token = normalized.split("-")[-1]
        if last_token:
            lookup[last_token] = canonical
            lookup[f"chunk-{last_token}"] = canonical
            lookup[f"-chunk-{last_token}"] = canonical

    return lookup


def _resolve_generated_chunk_id(raw_id: object, lookup: dict[str, str]) -> str | None:
    normalized = _normalize_chunk_id(raw_id)
    if not normalized:
        return None

    direct = lookup.get(normalized)
    if direct is not None:
        return direct

    lowered = normalized.lower()
    lowered_match = lookup.get(lowered)
    if lowered_match is not None:
        return lowered_match

    trimmed = lowered.replace(" ", "")
    return lookup.get(trimmed)


def build_instructor_client() -> instructor.Instructor:
    base_client = OpenAI(
        base_url=f"{OLLAMA_HOST}/v1",
        api_key=OLLAMA_API_KEY,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    return instructor.from_openai(base_client, mode=instructor.Mode.JSON)


def chat_structured(
    client: instructor.Instructor,
    response_model: type[BaseModel],
    user_prompt: str,
) -> BaseModel:
    logger.info("Calling LLM with structured response model: %s", response_model.__name__)
    return client.chat.completions.create(
        model=LLM,
        response_model=response_model,
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=LLM_TEMPERATURE,
        messages=[
            {
                "role": "system",
                "content": "You are a precise dataset benchmarking assistant. Return valid structured outputs only.",
            },
            {"role": "user", "content": user_prompt},
        ],
        extra_body={"options": {"num_ctx": NUM_CTX, "num_predict": MAX_OUTPUT_TOKENS}},
    )


def generate_user_tasks_questions(
    client: instructor.Instructor,
    total_description: str,
    user_index: int,
    existing_user_descriptions: list[str],
    allowed_chunk_ids: list[str],
) -> UserQuestionsResponse:
    chunk_label_to_id = {f"C{idx + 1}": chunk_id for idx, chunk_id in enumerate(allowed_chunk_ids)}
    allowed_chunk_labels = list(chunk_label_to_id.keys())
    allowed_chunk_label_text = "\n".join(
        f"- {label}: {chunk_id}" for label, chunk_id in chunk_label_to_id.items()
    )

    diversity_hint = ""
    if existing_user_descriptions:
        diversity_hint = (
            "\n\nPreviously generated user personas (avoid repeating these):\n"
            + "\n".join(f"- {desc}" for desc in existing_user_descriptions)
        )

    base_prompt = SIMPLE_USER_TASKS_QUESTIONS_PROMPT.format(
        total_description=total_description,
        tasks_per_user=TASKS_PER_USER,
        questions_per_task=QUESTIONS_PER_TASK,
        allowed_chunk_ids=allowed_chunk_label_text,
    )
    base_prompt = (
        f"{base_prompt}\n\nThis is generation #{user_index}. "
        "Create a distinct persona compared with previous generations."
        f"{diversity_hint}"
    )

    last_shape = "unknown"
    for attempt in range(1, MAX_GENERATION_RETRIES + 1):
        prompt = (
            f"{base_prompt}\n\n"
            "Hard requirement: return exactly "
            f"{TASKS_PER_USER} tasks and exactly {QUESTIONS_PER_TASK} questions per task. "
            "Do not return fewer items. "
            "Every question must include top_supporting_chunk_ids with 1 to 3 labels from the allowed set only (for example C1, C7, C12). "
            "Only use the short labels, not full chunk IDs. "
            "Every question must include a non-empty reference_answer."
        )
        if attempt > 1:
            prompt += (
                f"\n\nRetry attempt {attempt}/{MAX_GENERATION_RETRIES}. "
                f"Previous output shape was: {last_shape}."
            )

        try:
            response = chat_structured(client, UserQuestionsResponse, prompt)
        except IncompleteOutputException:
            logger.warning(
                "User %d generation attempt %d/%d truncated by max token limit; retrying",
                user_index,
                attempt,
                MAX_GENERATION_RETRIES,
            )
            last_shape = "incomplete_output_exception"
            continue

        raw_task_count = len(response.task_questions)
        raw_question_counts = [len(task.generated_questions) for task in response.task_questions]
        last_shape = f"tasks={raw_task_count}, questions_per_task={raw_question_counts}"

        if raw_task_count < TASKS_PER_USER:
            logger.warning(
                "User %d generation attempt %d/%d returned %d tasks (required %d)",
                user_index,
                attempt,
                MAX_GENERATION_RETRIES,
                raw_task_count,
                TASKS_PER_USER,
            )
            continue

        selected_tasks = response.task_questions[:TASKS_PER_USER]
        if any(len(task.generated_questions) < QUESTIONS_PER_TASK for task in selected_tasks):
            logger.warning(
                "User %d generation attempt %d/%d returned too few questions in at least one task: %s",
                user_index,
                attempt,
                MAX_GENERATION_RETRIES,
                raw_question_counts,
            )
            continue

        normalized_task_questions: list[TaskQuestions] = []
        allowed_chunk_id_set = set(allowed_chunk_ids)
        allowed_chunk_id_lookup = _build_allowed_chunk_id_lookup(allowed_chunk_ids)
        allowed_chunk_label_lookup = _build_allowed_chunk_id_lookup(allowed_chunk_labels)
        fallback_chunk_id = allowed_chunk_ids[0] if allowed_chunk_ids else None
        for task in selected_tasks:
            normalized_generated_questions: list[GeneratedQuestion] = []
            for generated_question in task.generated_questions[:QUESTIONS_PER_TASK]:
                filtered_ids: list[str] = []
                for chunk_id in generated_question.top_supporting_chunk_ids:
                    resolved_chunk_id = None

                    resolved_chunk_label = _resolve_generated_chunk_id(
                        chunk_id,
                        allowed_chunk_label_lookup,
                    )
                    if resolved_chunk_label is not None:
                        resolved_chunk_id = chunk_label_to_id.get(resolved_chunk_label)

                    if resolved_chunk_id is None:
                        resolved_chunk_id = _resolve_generated_chunk_id(chunk_id, allowed_chunk_id_lookup)

                    if (
                        resolved_chunk_id in allowed_chunk_id_set
                        and resolved_chunk_id not in filtered_ids
                    ):
                        filtered_ids.append(resolved_chunk_id)

                if not filtered_ids:
                    if fallback_chunk_id is None:
                        logger.warning(
                            "User %d generation attempt %d/%d returned question without valid supporting chunk IDs.",
                            user_index,
                            attempt,
                            MAX_GENERATION_RETRIES,
                        )
                        normalized_generated_questions = []
                        break

                    logger.warning(
                        "User %d generation attempt %d/%d returned unmapped supporting chunk IDs; using fallback chunk ID.",
                        user_index,
                        attempt,
                        MAX_GENERATION_RETRIES,
                    )
                    filtered_ids = [fallback_chunk_id]

                reference_answer = generated_question.reference_answer.strip()
                if not reference_answer:
                    logger.warning(
                        "User %d generation attempt %d/%d returned question without reference_answer.",
                        user_index,
                        attempt,
                        MAX_GENERATION_RETRIES,
                    )
                    normalized_generated_questions = []
                    break

                normalized_generated_questions.append(
                    GeneratedQuestion(
                        question=generated_question.question,
                        reference_answer=reference_answer,
                        top_supporting_chunk_ids=filtered_ids[:3],
                    )
                )

            if not normalized_generated_questions:
                normalized_task_questions = []
                break

            normalized_task_questions.append(
                TaskQuestions(
                    task_description=task.task_description,
                    generated_questions=normalized_generated_questions,
                )
            )

        if not normalized_task_questions:
            continue

        return UserQuestionsResponse(
            user_description=response.user_description,
            task_questions=normalized_task_questions,
        )

    raise ValueError(
        "Failed to generate exact task/question counts "
        f"for user {user_index} after {MAX_GENERATION_RETRIES} attempts. "
        f"Last output shape: {last_shape}"
    )


def run_dataset(dataset_name: str) -> None:
    output_dir = dataset_output_dir(dataset_name)

    output_dir.mkdir(parents=True, exist_ok=True)
    initialize_logger(output_dir, log_filename="step_2.log", verbose_debug=VERBOSE_DEBUG)
    logger.info("Starting dataset: %s", dataset_name)

    logger.info("Chunk store source: %s", chunk_store_file(dataset_name))
    user_chunk_pools = build_user_chunk_pools(dataset_name)
    logger.info("Prepared non-overlapping chunk pools for %d users", NUM_USERS)

    logger.info("Initializing instructor client")
    client = build_instructor_client()
    logger.info(
        "Generating %d users with %d tasks and %d questions per task",
        NUM_USERS,
        TASKS_PER_USER,
        QUESTIONS_PER_TASK,
    )

    users_output: list[UserOutput] = []
    existing_user_descriptions: list[str] = []
    for user_index in range(1, NUM_USERS + 1):
        logger.info("Generating user %d/%d", user_index, NUM_USERS)

        user_pool = user_chunk_pools[user_index - 1]
        sampled_chunks = sample_chunks_for_user(user_pool, dataset_name, user_index)
        logger.info("User %d sampled %d chunks", user_index, len(sampled_chunks))
        total_description = build_description_from_chunks(sampled_chunks)
        logger.info(
            "User %d description length: %d chars",
            user_index,
            len(total_description),
        )

        user_questions = generate_user_tasks_questions(
            client,
            total_description,
            user_index,
            existing_user_descriptions,
            [str(chunk["chunk_id"]) for chunk in sampled_chunks],
        )
        existing_user_descriptions.append(user_questions.user_description)

        users_output.append(
            UserOutput(
                user_description=user_questions.user_description,
                query=f"single_call_generation_user_{user_index}",
                sampled_chunks=sampled_chunks,
                task_questions=user_questions.task_questions,
            )
        )

        logger.info(
            "Completed user %d/%d (%d tasks generated)",
            user_index,
            NUM_USERS,
            len(user_questions.task_questions),
        )

    dataset_output = DatasetQuestionsOutput(dataset_name=dataset_name, users=users_output)

    output_file = output_dir / OUTPUT_FILE_NAME
    output_file.write_text(dataset_output.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Wrote questions output to: %s", output_file)

    metadata_file = output_dir / OUTPUT_METADATA_FILE_NAME
    metadata_payload = {
        "dataset_name": dataset_name,
        "random_seed": RANDOM_SEED,
        "step_1_experiment_name": STEP_1_EXPERIMENT_NAME,
        "chunk_store": str(chunk_store_file(dataset_name)),
        "chunks_per_dataset": CHUNKS_PER_DATASET,
        "sampling_mode": "per_user_seeded_consecutive_window",
        "user_chunk_pool_fraction": USER_CHUNK_POOL_FRACTION,
        "num_users": NUM_USERS,
        "tasks_per_user": TASKS_PER_USER,
        "questions_per_task": QUESTIONS_PER_TASK,
        "generation_mode": "single_call_user_tasks_questions_with_chunk_grounding",
    }
    metadata_file.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    logger.info("Wrote metadata output to: %s", metadata_file)

    print(f"[{dataset_name}] questions written to {output_file}")
    print(f"[{dataset_name}] chunk metadata written to {metadata_file}")
    logger.info("Completed dataset: %s", dataset_name)


def main() -> None:
    RESULT_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_name in DATASET_NAMES:
        output_dir = dataset_output_dir(dataset_name)
        constants_file = write_constants_snapshot(output_dir, globals())
        print(f"[{dataset_name}] constants snapshot written to: {constants_file}")
        run_dataset(dataset_name)


if __name__ == "__main__":
    main()
