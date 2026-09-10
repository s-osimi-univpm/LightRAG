from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

try:
    from .batch_eval_tie import (
        BaseModel,
        Field,
        OpenAI,
        is_event_loop_binding_error,
        instructor,
    )
except ImportError:  # Support direct execution: python better_reproduce/batch_eval.py
    from batch_eval_tie import (
        BaseModel,
        Field,
        OpenAI,
        is_event_loop_binding_error,
        instructor,
    )


# =========================
# Experiment constants
# =========================
DATASET_NAMES = ["agriculture"]#["legal","mix"]
EXPERIMENT_NAME = "rag_gemma31b_no_reason"

BASE_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = BASE_DIR / "results" / "3_rag"
QUESTIONS_RESULTS_ROOT = BASE_DIR / "results" / "2_questions"

OLLAMA_HOST = "http://localhost:11440"
EVAL_MODEL = "gemma4:31b"
NUM_CTX = 32768
MAX_EVAL_TOKENS = 2048
REQUEST_TIMEOUT_SECONDS = 1001
OLLAMA_API_KEY = "ollama"
LLM_TEMPERATURE = 0.2

INPUT_RAG_RESULTS_FILE_NAME = "rag_results.json"
RAG_RUN_SUMMARY_FILE_NAME = "rag_run_summary.json"
QUESTIONS_FILE_NAME = "questions.json"
OUTPUT_EVALUATION_NO_TIE_FILE_NAME = "evaluation_no_tie.json"
OUTPUT_SUMMARY_NO_TIE_FILE_NAME = "summary_no_tie.json"
# Filenames for additional pairwise comparisons involving MODE_3
OUTPUT_EVALUATION_NO_TIE_FILE_NAME_M1_M3 = "evaluation_no_tie_m1_m3.json"
OUTPUT_SUMMARY_NO_TIE_FILE_NAME_M1_M3 = "summary_no_tie_m1_m3.json"
OUTPUT_EVALUATION_NO_TIE_FILE_NAME_M2_M3 = "evaluation_no_tie_m2_m3.json"
OUTPUT_SUMMARY_NO_TIE_FILE_NAME_M2_M3 = "summary_no_tie_m2_m3.json"
RESUME_FROM_EXISTING_EVAL = True

MODE_1 = "hybrid"
MODE_2 = "naive"
MODE_3 = "mix"

NO_CONTEXT_PATTERNS = [
    "no context",
    "insufficient context",
    "not enough context",
    "context not provided",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "do not have enough information",
    "don't have enough information",
    "no relevant information",
]


class NoTieCriterionDecision(BaseModel):
    Winner: Literal["Answer 1", "Answer 2"]
    Explanation: str = Field(..., min_length=1)


class NoTieEvaluationResponse(BaseModel):
    Comprehensiveness: NoTieCriterionDecision
    Diversity: NoTieCriterionDecision
    Empowerment: NoTieCriterionDecision
    Overall_Winner: NoTieCriterionDecision = Field(alias="Overall Winner")


class ComparedAnswer(BaseModel):
    mode: str
    answer: str
    fundamental_chunk_count: int = 0
    retrieved_chunk_count: int = 0
    graph_hit_chunks: int = 0
    chunk_hit_count: int = 0
    graph_hit_count: int = 0
    retrieved_context_tokens: int = 0
    top_supporting_chunk_ids: list[str] = Field(default_factory=list)
    top3_supporting_chunk_ids: list[str] = Field(default_factory=list)
    top3_supporting_chunk_hit_count: int = 0
    top3_supporting_chunk_coverage: float | None = None


class QueryPair(BaseModel):
    question: str
    answer_1: ComparedAnswer
    answer_2: ComparedAnswer


class NoTieEvaluationRow(BaseModel):
    question: str
    answer_1_mode: str
    answer_2_mode: str
    answer_1_no_context: bool
    answer_2_no_context: bool
    answer_1_metrics: ComparedAnswer = Field(
        default_factory=lambda: ComparedAnswer(mode=MODE_1, answer="")
    )
    answer_2_metrics: ComparedAnswer = Field(
        default_factory=lambda: ComparedAnswer(mode=MODE_2, answer="")
    )
    evaluation: NoTieEvaluationResponse


class ModeTop3SupportingChunkStats(BaseModel):
    count: int
    top3_supporting_chunk_count_avg: float
    top3_supporting_chunk_hit_count_avg: float
    top3_supporting_chunk_coverage_avg: float


class NoTieSummary(BaseModel):
    total_questions: int
    no_context: dict[str, int]
    winner_label_mapping: dict[str, str]
    winners: dict[str, dict[str, int]]
    mode_top3_supporting_chunk_stats: dict[str, ModeTop3SupportingChunkStats]


def run_folder(dataset_name: str) -> Path:
    return RESULTS_ROOT / f"{dataset_name}_{EXPERIMENT_NAME}"


def rag_results_file(dataset_name: str) -> Path:
    return run_folder(dataset_name) / INPUT_RAG_RESULTS_FILE_NAME


def output_eval_file(dataset_name: str, filename: str = OUTPUT_EVALUATION_NO_TIE_FILE_NAME) -> Path:
    return run_folder(dataset_name) / filename


def output_summary_file(dataset_name: str, filename: str = OUTPUT_SUMMARY_NO_TIE_FILE_NAME) -> Path:
    return run_folder(dataset_name) / filename


def rag_run_summary_file(dataset_name: str) -> Path:
    return run_folder(dataset_name) / RAG_RUN_SUMMARY_FILE_NAME


def build_instructor_client() -> instructor.Instructor:
    base_client = OpenAI(
        base_url=f"{OLLAMA_HOST}/v1",
        api_key=OLLAMA_API_KEY,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    return instructor.from_openai(base_client, mode=instructor.Mode.JSON)


def has_no_context_signal(answer: str) -> bool:
    lowered = (answer or "").strip().lower()
    if not lowered:
        return True
    return any(pattern in lowered for pattern in NO_CONTEXT_PATTERNS)


def discover_questions_file(dataset_name: str) -> Path:
    summary_file = rag_run_summary_file(dataset_name)
    if summary_file.exists():
        payload = json.loads(summary_file.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            folder = str(payload.get("questions_folder", "")).strip()
            if folder:
                candidate = QUESTIONS_RESULTS_ROOT / folder / QUESTIONS_FILE_NAME
                if candidate.exists():
                    return candidate

    fallback_candidates = sorted(QUESTIONS_RESULTS_ROOT.glob(f"{dataset_name}_*/{QUESTIONS_FILE_NAME}"))
    if fallback_candidates:
        return fallback_candidates[-1]

    raise FileNotFoundError(
        f"Could not resolve questions file for dataset '{dataset_name}'. "
        f"Tried {summary_file} and fallback glob under {QUESTIONS_RESULTS_ROOT}."
    )


def build_top_supporting_chunk_lookup(
    dataset_name: str,
) -> tuple[dict[tuple[str, str, str], list[str]], dict[str, list[str]]]:
    questions_file = discover_questions_file(dataset_name)
    payload = json.loads(questions_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid questions payload in {questions_file}: expected dict")

    users = payload.get("users", [])
    if not isinstance(users, list):
        raise ValueError(f"Invalid users section in {questions_file}: expected list")

    by_triplet: dict[tuple[str, str, str], list[str]] = {}
    by_question_candidates: dict[str, list[list[str]]] = {}

    for user in users:
        if not isinstance(user, dict):
            continue
        user_description = str(user.get("user_description", "")).strip()
        task_questions = user.get("task_questions", [])
        if not isinstance(task_questions, list):
            continue

        for task in task_questions:
            if not isinstance(task, dict):
                continue
            task_description = str(task.get("task_description", "")).strip()
            generated_questions = task.get("generated_questions", [])
            if not isinstance(generated_questions, list):
                continue

            for generated in generated_questions:
                if not isinstance(generated, dict):
                    continue
                question = str(generated.get("question", "")).strip()
                if not question:
                    continue

                raw_ids = generated.get("top_supporting_chunk_ids", [])
                top_ids = [
                    str(chunk_id).strip()
                    for chunk_id in raw_ids
                    if str(chunk_id).strip()
                ] if isinstance(raw_ids, list) else []

                by_triplet[(user_description, task_description, question)] = top_ids
                by_question_candidates.setdefault(question, []).append(top_ids)

    by_question_unique: dict[str, list[str]] = {}
    for question, candidates in by_question_candidates.items():
        unique = {tuple(item) for item in candidates}
        if len(unique) == 1:
            by_question_unique[question] = list(unique.pop())

    return by_triplet, by_question_unique


def extract_retrieved_chunk_ids(row: dict[str, Any]) -> set[str]:
    retrieved_context = row.get("retrieved_context", {})
    if not isinstance(retrieved_context, dict):
        return set()

    chunks = retrieved_context.get("chunks", [])
    if not isinstance(chunks, list):
        return set()

    chunk_ids: set[str] = set()
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        if chunk_id:
            chunk_ids.add(chunk_id)

    # In hybrid mode, LightRAG can surface support via graph-linked source IDs
    # even when direct chunk lists are sparse.
    mode = str(row.get("mode", "")).strip().lower()
    if mode == MODE_1:
        for section_key in ("entities", "relationships"):
            section = retrieved_context.get(section_key, [])
            if not isinstance(section, list):
                continue
            for item in section:
                if not isinstance(item, dict):
                    continue
                source_id = item.get("source_id", "")
                if not isinstance(source_id, str):
                    continue
                for raw_chunk_id in source_id.split("<SEP>"):
                    chunk_id = raw_chunk_id.strip()
                    if chunk_id:
                        chunk_ids.add(chunk_id)

    return chunk_ids


def resolve_top_supporting_chunk_ids(
    row: dict[str, Any],
    by_triplet: dict[tuple[str, str, str], list[str]],
    by_question_unique: dict[str, list[str]],
) -> list[str]:
    user_description = str(row.get("user_description", "")).strip()
    task_description = str(row.get("task_description", "")).strip()
    question = str(row.get("question", "")).strip()

    top_ids = by_triplet.get((user_description, task_description, question))
    if top_ids is not None:
        return top_ids

    return by_question_unique.get(question, [])


def compute_top3_supporting_chunk_metrics(
    top_supporting_chunk_ids: list[str],
    retrieved_chunk_ids: set[str],
) -> tuple[list[str], int, float | None]:
    deduplicated_top3: list[str] = []
    for chunk_id in top_supporting_chunk_ids:
        normalized = chunk_id.strip()
        if normalized and normalized not in deduplicated_top3:
            deduplicated_top3.append(normalized)
        if len(deduplicated_top3) == 3:
            break

    if not deduplicated_top3:
        return [], 0, None

    hit_count = sum(1 for chunk_id in deduplicated_top3 if chunk_id in retrieved_chunk_ids)
    coverage = hit_count / len(deduplicated_top3)
    return deduplicated_top3, hit_count, coverage


def load_query_pairs_from_rag_results(
    file_path: Path,
    dataset_name: str,
    mode_a: str = MODE_1,
    mode_b: str = MODE_2,
) -> list[QueryPair]:
    if not file_path.exists():
        raise FileNotFoundError(f"RAG results file not found: {file_path}")

    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {file_path}")

    top_supporting_by_triplet, top_supporting_by_question = build_top_supporting_chunk_lookup(
        dataset_name
    )

    mode_1_rows = [row for row in payload if isinstance(row, dict) and row.get("mode") == mode_a]
    mode_2_rows = [row for row in payload if isinstance(row, dict) and row.get("mode") == mode_b]

    if len(mode_1_rows) != len(mode_2_rows):
        raise ValueError(
            f"Mode size mismatch in {file_path}: "
            f"{mode_a}={len(mode_1_rows)} {mode_b}={len(mode_2_rows)}"
        )

    pairs: list[QueryPair] = []
    for idx, (row_1, row_2) in enumerate(zip(mode_1_rows, mode_2_rows), start=1):
        question_1 = str(row_1.get("question", "")).strip()
        question_2 = str(row_2.get("question", "")).strip()
        if question_1 != question_2:
            raise ValueError(
                f"Question mismatch at pair #{idx} in {file_path}: "
                f"{mode_a} question differs from {mode_b} question"
            )

        top_supporting_ids = resolve_top_supporting_chunk_ids(
            row_1,
            top_supporting_by_triplet,
            top_supporting_by_question,
        )
        retrieved_chunk_ids_1 = extract_retrieved_chunk_ids(row_1)
        retrieved_chunk_ids_2 = extract_retrieved_chunk_ids(row_2)

        top3_ids_1, top3_hit_count_1, top3_coverage_1 = compute_top3_supporting_chunk_metrics(
            top_supporting_ids,
            retrieved_chunk_ids_1,
        )
        top3_ids_2, top3_hit_count_2, top3_coverage_2 = compute_top3_supporting_chunk_metrics(
            top_supporting_ids,
            retrieved_chunk_ids_2,
        )

        pairs.append(
            QueryPair(
                question=question_1,
                answer_1=ComparedAnswer(
                    mode=mode_a,
                    answer=str(row_1.get("raw_result", "")),
                    fundamental_chunk_count=int(row_1.get("fundamental_chunk_count", 0) or 0),
                    retrieved_chunk_count=int(row_1.get("retrieved_chunk_count", 0) or 0),
                    graph_hit_chunks=int(row_1.get("graph_hit_chunks", 0) or 0),
                    chunk_hit_count=int(row_1.get("chunk_hit_count", 0) or 0),
                    graph_hit_count=int(row_1.get("graph_hit_count", 0) or 0),
                    retrieved_context_tokens=int(row_1.get("retrieved_context_tokens", 0) or 0),
                    top_supporting_chunk_ids=top_supporting_ids,
                    top3_supporting_chunk_ids=top3_ids_1,
                    top3_supporting_chunk_hit_count=top3_hit_count_1,
                    top3_supporting_chunk_coverage=top3_coverage_1,
                ),
                answer_2=ComparedAnswer(
                    mode=mode_b,
                    answer=str(row_2.get("raw_result", "")),
                    fundamental_chunk_count=int(row_2.get("fundamental_chunk_count", 0) or 0),
                    retrieved_chunk_count=int(row_2.get("retrieved_chunk_count", 0) or 0),
                    graph_hit_chunks=int(row_2.get("graph_hit_chunks", 0) or 0),
                    chunk_hit_count=int(row_2.get("chunk_hit_count", 0) or 0),
                    graph_hit_count=int(row_2.get("graph_hit_count", 0) or 0),
                    retrieved_context_tokens=int(row_2.get("retrieved_context_tokens", 0) or 0),
                    top_supporting_chunk_ids=top_supporting_ids,
                    top3_supporting_chunk_ids=top3_ids_2,
                    top3_supporting_chunk_hit_count=top3_hit_count_2,
                    top3_supporting_chunk_coverage=top3_coverage_2,
                ),
            )
        )

    return pairs


def evaluate_pair_no_tie(client, pair, no_context_1: bool, no_context_2: bool) -> NoTieEvaluationResponse:
    prompt = f"""
---Role---
You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

---Goal---
You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

- **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
- **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
- **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.

Here is the question:
{pair.question}

Here are the two answers:

**Answer 1 ({pair.answer_1.mode}, no_context_signal={no_context_1}):**
{pair.answer_1.answer}

**Answer 2 ({pair.answer_2.mode}, no_context_signal={no_context_2}):**
{pair.answer_2.answer}

Evaluate both answers using the three criteria listed above and provide concise explanations for each criterion (at most 2 sentences each).

IMPORTANT: You MUST return a valid JSON object with EXACTLY these four keys at the top level:
- "Comprehensiveness" (object with "Winner" and "Explanation" keys)
- "Diversity" (object with "Winner" and "Explanation" keys)  
- "Empowerment" (object with "Winner" and "Explanation" keys)
- "Overall Winner" (object with "Winner" and "Explanation" keys)

Each criterion object MUST have exactly two keys:
1. "Winner": either "Answer 1" or "Answer 2" (exactly as written)
2. "Explanation": a string with your reasoning (2 sentences max)

Output your evaluation in the following JSON format:

{{
    "Comprehensiveness": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Diversity": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Empowerment": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Overall Winner": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"
    }}
}}
""".strip()

    raw = client.chat.completions.create(
        model=EVAL_MODEL,
        response_model=NoTieEvaluationResponse,
        max_tokens=MAX_EVAL_TOKENS,
        temperature=LLM_TEMPERATURE,
        messages=[
            {
                "role": "system",
                "content": "You are a strict evaluation judge. Return VALID JSON only. Every object must have 'Winner' and 'Explanation' keys.",
            },
            {"role": "user", "content": prompt},
        ],
        extra_body={"options": {"num_ctx": NUM_CTX}},
    )
    return NoTieEvaluationResponse.model_validate(raw.model_dump(by_alias=True))


def is_incomplete_output_error(exc: Exception) -> bool:
    message = str(exc).lower()
    class_name = exc.__class__.__name__.lower()
    return "incompleteoutputexception" in class_name or "output is incomplete" in message


def is_validation_error(exc: Exception) -> bool:
    """Check if the error is a Pydantic validation error from the LLM output."""
    message = str(exc).lower()
    class_name = exc.__class__.__name__.lower()
    return "validationerror" in class_name or "field required" in message


def summarize_no_tie(rows: list[NoTieEvaluationRow], mode_a: str = MODE_1, mode_b: str = MODE_2) -> NoTieSummary:
    winners = {
        "Comprehensiveness": {"Answer 1": 0, "Answer 2": 0},
        "Diversity": {"Answer 1": 0, "Answer 2": 0},
        "Empowerment": {"Answer 1": 0, "Answer 2": 0},
        "Overall Winner": {"Answer 1": 0, "Answer 2": 0},
    }
    no_context = {
        "answer_1": 0,
        "answer_2": 0,
        "both": 0,
        "either": 0,
        "neither": 0,
    }
    top3_stats_acc: dict[str, dict[str, float | int]] = {}

    def ensure_mode(mode: str) -> None:
        if mode in top3_stats_acc:
            return
        top3_stats_acc[mode] = {
            "count": 0,
            "top3_count_sum": 0,
            "top3_hit_sum": 0,
            "top3_coverage_sum": 0.0,
            "top3_coverage_count": 0,
        }

    def accumulate_top3(answer: ComparedAnswer) -> None:
        ensure_mode(answer.mode)
        acc = top3_stats_acc[answer.mode]
        acc["count"] = int(acc["count"]) + 1
        acc["top3_count_sum"] = int(acc["top3_count_sum"]) + len(answer.top3_supporting_chunk_ids)
        acc["top3_hit_sum"] = int(acc["top3_hit_sum"]) + answer.top3_supporting_chunk_hit_count
        if answer.top3_supporting_chunk_coverage is not None:
            acc["top3_coverage_sum"] = float(acc["top3_coverage_sum"]) + answer.top3_supporting_chunk_coverage
            acc["top3_coverage_count"] = int(acc["top3_coverage_count"]) + 1

    for row in rows:
        winners["Comprehensiveness"][row.evaluation.Comprehensiveness.Winner] += 1
        winners["Diversity"][row.evaluation.Diversity.Winner] += 1
        winners["Empowerment"][row.evaluation.Empowerment.Winner] += 1
        winners["Overall Winner"][row.evaluation.Overall_Winner.Winner] += 1
        accumulate_top3(row.answer_1_metrics)
        accumulate_top3(row.answer_2_metrics)

        if row.answer_1_no_context:
            no_context["answer_1"] += 1
        if row.answer_2_no_context:
            no_context["answer_2"] += 1
        if row.answer_1_no_context and row.answer_2_no_context:
            no_context["both"] += 1
        elif row.answer_1_no_context or row.answer_2_no_context:
            no_context["either"] += 1
        else:
            no_context["neither"] += 1

    mode_top3_stats: dict[str, ModeTop3SupportingChunkStats] = {}
    for mode, acc in top3_stats_acc.items():
        count = int(acc["count"])
        if count == 0:
            continue
        coverage_count = int(acc["top3_coverage_count"])
        mode_top3_stats[mode] = ModeTop3SupportingChunkStats(
            count=count,
            top3_supporting_chunk_count_avg=int(acc["top3_count_sum"]) / count,
            top3_supporting_chunk_hit_count_avg=int(acc["top3_hit_sum"]) / count,
            top3_supporting_chunk_coverage_avg=(
                float(acc["top3_coverage_sum"]) / coverage_count
                if coverage_count > 0
                else 0.0
            ),
        )

    return NoTieSummary(
        total_questions=len(rows),
        no_context=no_context,
        winner_label_mapping={"Answer 1": mode_a, "Answer 2": mode_b},
        winners=winners,
        mode_top3_supporting_chunk_stats=mode_top3_stats,
    )


def save_progress(
    eval_file: Path,
    summary_file: Path,
    rows: list[NoTieEvaluationRow],
    mode_a: str = MODE_1,
    mode_b: str = MODE_2,
) -> None:
    summary = summarize_no_tie(rows, mode_a, mode_b)
    eval_file.write_text(
        json.dumps([row.model_dump(by_alias=True) for row in rows], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary_file.write_text(summary.model_dump_json(indent=2), encoding="utf-8")


def evaluate_dataset_no_tie(
    dataset_name: str,
    mode_a: str = MODE_1,
    mode_b: str = MODE_2,
    eval_filename: str = OUTPUT_EVALUATION_NO_TIE_FILE_NAME,
    summary_filename: str = OUTPUT_SUMMARY_NO_TIE_FILE_NAME,
) -> None:
    pairs = load_query_pairs_from_rag_results(rag_results_file(dataset_name), dataset_name, mode_a, mode_b)
    client = build_instructor_client()
    eval_file = output_eval_file(dataset_name, eval_filename)
    summary_file = output_summary_file(dataset_name, summary_filename)

    rows: list[NoTieEvaluationRow] = []
    if RESUME_FROM_EXISTING_EVAL and eval_file.exists():
        existing_payload = json.loads(eval_file.read_text(encoding="utf-8"))
        if not isinstance(existing_payload, list):
            raise ValueError(f"Expected list in {eval_file}")

        rows = [NoTieEvaluationRow.model_validate(item) for item in existing_payload]

        if len(rows) > len(pairs):
            raise ValueError(
                f"Existing evaluation has more rows than input pairs in {dataset_name}: "
                f"rows={len(rows)} pairs={len(pairs)}"
            )

        if rows:
            # Refresh stored retrieval metrics using current metric logic while
            # preserving previous judge outputs.
            refreshed_rows: list[NoTieEvaluationRow] = []
            for idx, (row, pair) in enumerate(zip(rows, pairs), start=1):
                if row.question != pair.question:
                    raise ValueError(
                        f"Question mismatch while refreshing existing evaluation in {dataset_name} "
                        f"at row #{idx}: '{row.question}' != '{pair.question}'"
                    )

                refreshed_rows.append(
                    row.model_copy(
                        update={
                            "answer_1_mode": pair.answer_1.mode,
                            "answer_2_mode": pair.answer_2.mode,
                            "answer_1_no_context": has_no_context_signal(pair.answer_1.answer),
                            "answer_2_no_context": has_no_context_signal(pair.answer_2.answer),
                            "answer_1_metrics": pair.answer_1,
                            "answer_2_metrics": pair.answer_2,
                        }
                    )
                )
            rows = refreshed_rows

            print(
                f"[{dataset_name}] Resuming from existing evaluation file; "
                f"skipping {len(rows)} completed questions."
            )
            save_progress(eval_file, summary_file, rows, mode_a, mode_b)

    start_idx = len(rows)
    for idx, pair in enumerate(pairs[start_idx:], start=start_idx + 1):
        print(f"[{dataset_name}] Evaluating question {idx}/{len(pairs)}")
        no_context_1 = has_no_context_signal(pair.answer_1.answer)
        no_context_2 = has_no_context_signal(pair.answer_2.answer)

        try:
            evaluation = evaluate_pair_no_tie(client, pair, no_context_1, no_context_2)
        except Exception as exc:  # noqa: BLE001
            if not (is_event_loop_binding_error(exc) or is_incomplete_output_error(exc) or is_validation_error(exc)):
                raise

            if is_event_loop_binding_error(exc):
                print(
                    f"[{dataset_name}] Event-loop binding error detected while evaluating question {idx}; "
                    "rebuilding client and retrying once."
                )
            elif is_incomplete_output_error(exc):
                print(
                    f"[{dataset_name}] Incomplete model output detected while evaluating question {idx}; "
                    "rebuilding client and retrying once."
                )
            else:
                print(
                    f"[{dataset_name}] JSON validation error detected while evaluating question {idx}; "
                    "rebuilding client and retrying once. Error: {str(exc)[:200]}"
                )

            client = build_instructor_client()
            evaluation = evaluate_pair_no_tie(client, pair, no_context_1, no_context_2)

        rows.append(
            NoTieEvaluationRow(
                question=pair.question,
                answer_1_mode=pair.answer_1.mode,
                answer_2_mode=pair.answer_2.mode,
                answer_1_no_context=no_context_1,
                answer_2_no_context=no_context_2,
                answer_1_metrics=pair.answer_1,
                answer_2_metrics=pair.answer_2,
                evaluation=evaluation,
            )
        )
        save_progress(eval_file, summary_file, rows, mode_a, mode_b)

    print(f"[{dataset_name}] evaluation saved to {eval_file}")
    print(f"[{dataset_name}] summary saved to {summary_file}")


def main() -> None:
    for dataset_name in DATASET_NAMES:
        # naive vs hybrid
        evaluate_dataset_no_tie(
            dataset_name,
            mode_a=MODE_2,
            mode_b=MODE_1,
        )
        # naive vs mix
        evaluate_dataset_no_tie(
            dataset_name,
            mode_a=MODE_2,
            mode_b=MODE_3,
            eval_filename=OUTPUT_EVALUATION_NO_TIE_FILE_NAME_M2_M3,
            summary_filename=OUTPUT_SUMMARY_NO_TIE_FILE_NAME_M2_M3,
        )


if __name__ == "__main__":
    main()
