from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field


# =========================
# Experiment constants
# =========================
DATASET_NAMES = ["agriculture","legal"]
EXPERIMENT_NAME = "rag_gemma4b"

BASE_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = BASE_DIR / "results" / "3_rag"

OLLAMA_HOST = "http://localhost:11440"
EVAL_MODEL = "gemma4:31b"
NUM_CTX = 32768
REQUEST_TIMEOUT_SECONDS = 1001
OLLAMA_API_KEY = "ollama"
LLM_TEMPERATURE = 0.2

INPUT_RAG_RESULTS_FILE_NAME = "rag_results.json"
OUTPUT_EVALUATION_FILE_NAME = "evaluation.json"
OUTPUT_SUMMARY_FILE_NAME = "summary.json"
OUTPUT_EVALUATION_NO_TIE_FILE_NAME = "evaluation_no_tie.json"
OUTPUT_SUMMARY_NO_TIE_FILE_NAME = "summary_no_tie.json"

MODE_1 = "hybrid"
MODE_2 = "naive"
ALLOW_TIES_BY_DEFAULT = True

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


class ComparedAnswer(BaseModel):
    mode: str
    answer: str
    sampled_chunk_count: int = 0
    retrieved_chunk_count: int = 0
    chunk_hit_rate: float = 0.0
    entity_linked_hit_count: int = 0
    entity_contribution_index: float | None = None
    retrieved_context_tokens: int = 0


class QueryPair(BaseModel):
    question: str
    answer_1: ComparedAnswer
    answer_2: ComparedAnswer


class CriterionDecision(BaseModel):
    Winner: Literal["Answer 1", "Answer 2", "Tie"]
    Explanation: str = Field(..., min_length=1)


class TieDecision(BaseModel):
    is_tie: bool
    tie_confidence: float = Field(..., ge=0.0, le=1.0)
    tie_reason: str = Field(..., min_length=1)


class EvaluationResponse(BaseModel):
    Comprehensiveness: CriterionDecision
    Diversity: CriterionDecision
    Empowerment: CriterionDecision
    Overall_Winner: CriterionDecision = Field(alias="Overall Winner")
    tie_decision: TieDecision
    final_winner: Literal["Answer 1", "Answer 2"]
    final_winner_reason: str = Field(..., min_length=1)


class EvaluationRow(BaseModel):
    question: str
    answer_1_mode: str
    answer_2_mode: str
    answer_1_no_context: bool
    answer_2_no_context: bool
    answer_1_metrics: ComparedAnswer
    answer_2_metrics: ComparedAnswer
    evaluation: EvaluationResponse


class ModeRetrievalStats(BaseModel):
    count: int
    sampled_chunk_count_avg: float
    retrieved_chunk_count_avg: float
    chunk_hit_rate_avg: float
    entity_linked_hit_count_avg: float
    entity_contribution_index_avg: float | None
    retrieved_context_tokens_avg: float


class EvaluationSummary(BaseModel):
    total_questions: int
    no_context: dict[str, int]
    winner_label_mapping: dict[str, str]
    winners: dict[str, dict[str, int]]
    final_winner_counts: dict[str, int]
    mode_retrieval_stats: dict[str, ModeRetrievalStats]


def run_folder(dataset_name: str) -> Path:
    return RESULTS_ROOT / f"{dataset_name}_{EXPERIMENT_NAME}"


def rag_results_file(dataset_name: str) -> Path:
    return run_folder(dataset_name) / INPUT_RAG_RESULTS_FILE_NAME


def output_eval_file(dataset_name: str, allow_tie: bool) -> Path:
    file_name = OUTPUT_EVALUATION_FILE_NAME if allow_tie else OUTPUT_EVALUATION_NO_TIE_FILE_NAME
    return run_folder(dataset_name) / file_name


def output_summary_file(dataset_name: str, allow_tie: bool) -> Path:
    file_name = OUTPUT_SUMMARY_FILE_NAME if allow_tie else OUTPUT_SUMMARY_NO_TIE_FILE_NAME
    return run_folder(dataset_name) / file_name


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


def load_query_pairs_from_rag_results(file_path: Path) -> list[QueryPair]:
    if not file_path.exists():
        raise FileNotFoundError(f"RAG results file not found: {file_path}")

    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {file_path}")

    mode_1_rows = [row for row in payload if isinstance(row, dict) and row.get("mode") == MODE_1]
    mode_2_rows = [row for row in payload if isinstance(row, dict) and row.get("mode") == MODE_2]

    if len(mode_1_rows) != len(mode_2_rows):
        raise ValueError(
            f"Mode size mismatch in {file_path}: "
            f"{MODE_1}={len(mode_1_rows)} {MODE_2}={len(mode_2_rows)}"
        )

    pairs: list[QueryPair] = []
    for idx, (row_1, row_2) in enumerate(zip(mode_1_rows, mode_2_rows), start=1):
        question_1 = str(row_1.get("question", "")).strip()
        question_2 = str(row_2.get("question", "")).strip()
        if question_1 != question_2:
            raise ValueError(
                f"Question mismatch at pair #{idx} in {file_path}: "
                f"{MODE_1} question differs from {MODE_2} question"
            )

        answer_1 = str(row_1.get("raw_result", ""))
        answer_2 = str(row_2.get("raw_result", ""))

        metrics_1 = ComparedAnswer(
            mode=MODE_1,
            answer=answer_1,
            sampled_chunk_count=int(row_1.get("sampled_chunk_count", 0) or 0),
            retrieved_chunk_count=int(row_1.get("retrieved_chunk_count", 0) or 0),
            chunk_hit_rate=float(row_1.get("chunk_hit_rate", 0.0) or 0.0),
            entity_linked_hit_count=int(row_1.get("entity_linked_hit_count", 0) or 0),
            entity_contribution_index=(
                float(row_1.get("entity_contribution_index"))
                if row_1.get("entity_contribution_index") is not None
                else None
            ),
            retrieved_context_tokens=int(row_1.get("retrieved_context_tokens", 0) or 0),
        )
        metrics_2 = ComparedAnswer(
            mode=MODE_2,
            answer=answer_2,
            sampled_chunk_count=int(row_2.get("sampled_chunk_count", 0) or 0),
            retrieved_chunk_count=int(row_2.get("retrieved_chunk_count", 0) or 0),
            chunk_hit_rate=float(row_2.get("chunk_hit_rate", 0.0) or 0.0),
            entity_linked_hit_count=int(row_2.get("entity_linked_hit_count", 0) or 0),
            entity_contribution_index=(
                float(row_2.get("entity_contribution_index"))
                if row_2.get("entity_contribution_index") is not None
                else None
            ),
            retrieved_context_tokens=int(row_2.get("retrieved_context_tokens", 0) or 0),
        )

        pairs.append(
            QueryPair(
                question=question_1,
                answer_1=metrics_1,
                answer_2=metrics_2,
            )
        )

    return pairs


def evaluate_pair(
    client: instructor.Instructor,
    pair: QueryPair,
    no_context_1: bool,
    no_context_2: bool,
    allow_tie: bool,
) -> EvaluationResponse:
    tie_mode_rules = (
        "- Choose a Winner for each criterion: Answer 1, Answer 2, or Tie.\n"
        "- Choose an overall winner with the same allowed labels.\n"
        "- Compute tie_decision normally (is_tie may be true or false)."
        if allow_tie
        else "- Choose a Winner for each criterion: Answer 1 or Answer 2 only (no Tie).\n"
        "- Choose an overall winner: Answer 1 or Answer 2 only (no Tie).\n"
        "- Set tie_decision.is_tie=false with tie_confidence=0 and tie_reason='Ties disabled by evaluator setting'."
    )

    prompt = f"""
You are evaluating two answers to the same question.

Task:
- Step 1: Decide whether this comparison is a Tie.
- Step 2: Always choose a final winner (Answer 1 or Answer 2), even if Step 1 is Tie.

Criteria:
- Comprehensiveness
- Diversity
- Empowerment
- Overall quality

Rules:
- {tie_mode_rules}
- Compute tie_decision:
    - is_tie=true only if ties are enabled and the two answers are materially equivalent across criteria.
    - tie_confidence in [0, 1].
    - tie_reason should be concise.
- Always output final_winner as Answer 1 or Answer 2.
- If tie_decision.is_tie=true, break ties using this deterministic order:
    1) Better factual grounding to the question
    2) Higher directness and specificity
    3) Fewer unsupported claims
    4) Better structure and clarity
    5) If still equal, choose Answer 1
- Consider factual usefulness and directness.
- If one answer says there is no context, this is still a valid answer candidate and must still be evaluated.
- Respond only with valid JSON matching the required schema.

Question:
{pair.question}

Answer 1 ({pair.answer_1.mode}) | no_context_signal={no_context_1}:
{pair.answer_1.answer}

Answer 2 ({pair.answer_2.mode}) | no_context_signal={no_context_2}:
{pair.answer_2.answer}
""".strip()

    raw = client.chat.completions.create(
        model=EVAL_MODEL,
        response_model=EvaluationResponse,
        temperature=LLM_TEMPERATURE,
        messages=[
            {
                "role": "system",
                "content": "You are a strict evaluation judge. Return valid structured outputs only.",
            },
            {"role": "user", "content": prompt},
        ],
        extra_body={"options": {"num_ctx": NUM_CTX}},
    )
    return EvaluationResponse.model_validate(raw.model_dump(by_alias=True))


def is_event_loop_binding_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "bound to a different event loop" in message or "different event loop" in message


def summarize(rows: list[EvaluationRow], allow_tie: bool) -> EvaluationSummary:
    winner_labels = ["Answer 1", "Answer 2", "Tie"] if allow_tie else ["Answer 1", "Answer 2"]
    winners = {
        "Comprehensiveness": {label: 0 for label in winner_labels},
        "Diversity": {label: 0 for label in winner_labels},
        "Empowerment": {label: 0 for label in winner_labels},
        "Overall Winner": {label: 0 for label in winner_labels},
    }
    final_winner_counts = {"Answer 1": 0, "Answer 2": 0}

    stats_acc: dict[str, dict[str, float | int]] = {}

    def ensure_mode(mode: str) -> None:
        if mode in stats_acc:
            return
        stats_acc[mode] = {
            "count": 0,
            "sampled_sum": 0,
            "retrieved_sum": 0,
            "hit_rate_sum": 0.0,
            "entity_linked_sum": 0,
            "entity_contrib_sum": 0.0,
            "entity_contrib_count": 0,
            "tokens_sum": 0,
        }

    def accumulate(answer: ComparedAnswer) -> None:
        ensure_mode(answer.mode)
        acc = stats_acc[answer.mode]
        acc["count"] = int(acc["count"]) + 1
        acc["sampled_sum"] = int(acc["sampled_sum"]) + answer.sampled_chunk_count
        acc["retrieved_sum"] = int(acc["retrieved_sum"]) + answer.retrieved_chunk_count
        acc["hit_rate_sum"] = float(acc["hit_rate_sum"]) + answer.chunk_hit_rate
        acc["entity_linked_sum"] = int(acc["entity_linked_sum"]) + answer.entity_linked_hit_count
        acc["tokens_sum"] = int(acc["tokens_sum"]) + answer.retrieved_context_tokens
        if answer.entity_contribution_index is not None:
            acc["entity_contrib_sum"] = float(acc["entity_contrib_sum"]) + answer.entity_contribution_index
            acc["entity_contrib_count"] = int(acc["entity_contrib_count"]) + 1

    no_context = {
        "answer_1": 0,
        "answer_2": 0,
        "both": 0,
        "either": 0,
        "neither": 0,
    }

    for row in rows:
        comprehensiveness_winner = row.evaluation.Comprehensiveness.Winner
        diversity_winner = row.evaluation.Diversity.Winner
        empowerment_winner = row.evaluation.Empowerment.Winner
        overall_winner = row.evaluation.Overall_Winner.Winner

        if comprehensiveness_winner not in winners["Comprehensiveness"]:
            winners["Comprehensiveness"][comprehensiveness_winner] = 0
        if diversity_winner not in winners["Diversity"]:
            winners["Diversity"][diversity_winner] = 0
        if empowerment_winner not in winners["Empowerment"]:
            winners["Empowerment"][empowerment_winner] = 0
        if overall_winner not in winners["Overall Winner"]:
            winners["Overall Winner"][overall_winner] = 0

        winners["Comprehensiveness"][comprehensiveness_winner] += 1
        winners["Diversity"][diversity_winner] += 1
        winners["Empowerment"][empowerment_winner] += 1
        winners["Overall Winner"][overall_winner] += 1
        final_winner_counts[row.evaluation.final_winner] += 1

        accumulate(row.answer_1_metrics)
        accumulate(row.answer_2_metrics)

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

    mode_retrieval_stats: dict[str, ModeRetrievalStats] = {}
    for mode, acc in stats_acc.items():
        count = int(acc["count"])
        if count == 0:
            continue

        entity_contrib_count = int(acc["entity_contrib_count"])
        entity_contribution_index_avg: float | None = None
        if entity_contrib_count > 0:
            entity_contribution_index_avg = float(acc["entity_contrib_sum"]) / entity_contrib_count

        mode_retrieval_stats[mode] = ModeRetrievalStats(
            count=count,
            sampled_chunk_count_avg=int(acc["sampled_sum"]) / count,
            retrieved_chunk_count_avg=int(acc["retrieved_sum"]) / count,
            chunk_hit_rate_avg=float(acc["hit_rate_sum"]) / count,
            entity_linked_hit_count_avg=int(acc["entity_linked_sum"]) / count,
            entity_contribution_index_avg=entity_contribution_index_avg,
            retrieved_context_tokens_avg=int(acc["tokens_sum"]) / count,
        )

    return EvaluationSummary(
        total_questions=len(rows),
        no_context=no_context,
        winner_label_mapping={"Answer 1": MODE_1, "Answer 2": MODE_2},
        winners=winners,
        final_winner_counts=final_winner_counts,
        mode_retrieval_stats=mode_retrieval_stats,
    )


def evaluate_dataset(dataset_name: str, allow_tie: bool) -> None:
    input_file = rag_results_file(dataset_name)
    eval_file = output_eval_file(dataset_name, allow_tie)
    summary_file = output_summary_file(dataset_name, allow_tie)

    pairs = load_query_pairs_from_rag_results(input_file)
    client = build_instructor_client()

    rows: list[EvaluationRow] = []
    for idx, pair in enumerate(pairs, start=1):
        print(f"[{dataset_name}] Evaluating question {idx}/{len(pairs)}")

        no_context_1 = has_no_context_signal(pair.answer_1.answer)
        no_context_2 = has_no_context_signal(pair.answer_2.answer)

        try:
            evaluation = evaluate_pair(client, pair, no_context_1, no_context_2, allow_tie)
        except Exception as exc:  # noqa: BLE001
            if not is_event_loop_binding_error(exc):
                raise

            print(
                f"[{dataset_name}] Event-loop binding error detected while evaluating question {idx}; "
                "rebuilding client and retrying once."
            )
            client = build_instructor_client()
            evaluation = evaluate_pair(client, pair, no_context_1, no_context_2, allow_tie)

        rows.append(
            EvaluationRow(
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

    summary = summarize(rows, allow_tie)

    eval_file.write_text(
        json.dumps([row.model_dump(by_alias=True) for row in rows], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary_file.write_text(
        summary.model_dump_json(indent=2),
        encoding="utf-8",
    )

    print(f"[{dataset_name}] evaluation saved to {eval_file}")
    print(f"[{dataset_name}] summary saved to {summary_file}")


def main(force_no_tie: bool | None = None) -> None:
    if force_no_tie is None:
        parser = argparse.ArgumentParser(description="Batch evaluate RAG answers.")
        parser.add_argument(
            "--no-tie",
            action="store_true",
            help="Disallow tie decisions and write *_no_tie outputs.",
        )
        args = parser.parse_args()
        allow_tie = ALLOW_TIES_BY_DEFAULT
        if args.no_tie:
            allow_tie = False
    else:
        allow_tie = not force_no_tie

    for dataset_name in DATASET_NAMES:
        evaluate_dataset(dataset_name, allow_tie)


if __name__ == "__main__":
    main()