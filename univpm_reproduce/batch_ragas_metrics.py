from __future__ import annotations

import argparse
import json
import math
import os
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

# =========================
# Experiment constants
# =========================
DATASET_NAMES = ["agriculture", "legal", "mix"]
EXPERIMENT_NAME = "rag_gemma31b_no_reason"

BASE_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = BASE_DIR / "results" / "3_rag"

INPUT_RAG_RESULTS_FILE_NAME = "rag_results.json"
OUTPUT_RAGAS_DETAILS_FILE_NAME = "ragas_metrics.json"
OUTPUT_RAGAS_SUMMARY_FILE_NAME = "ragas_summary.json"
OUTPUT_RAGAS_INTERMEDIATE_FILE_NAME = "ragas_intermediate.json"

# OpenAI-compatible endpoint for evaluator model (Ollama/vLLM/OpenAI-compatible).
OLLAMA_HOST = "http://localhost:11440"
EVAL_MODEL = "gemma4:31b"
REQUEST_TIMEOUT_SECONDS = 1200
OLLAMA_API_KEY = "ollama"

# Embedding strategy aligned with univpm_reproduce/1_indexing.py:
# local SentenceTransformer model on a configured device.
EVAL_EMBEDDING_MODEL = os.getenv("RAGAS_EVAL_EMBEDDING_MODEL", "BAAI/bge-m3")
EVAL_EMBEDDING_DEVICE = os.getenv("RAGAS_EVAL_EMBEDDING_DEVICE", "cuda:1")
EVAL_EMBED_BATCH_SIZE = int(os.getenv("RAGAS_EVAL_EMBED_BATCH_SIZE", "512"))

# Runtime behavior
SAVE_EVERY_SAMPLES = 1
STRICTNESS = 3
METRIC_TIMEOUT_BUFFER_SECONDS = 5


class MetricTimeoutError(RuntimeError):
    pass


class MetricTrace(BaseModel):
    score: float | None
    reason: str | None = None
    error: str | None = None
    intermediate: dict[str, Any] = Field(default_factory=dict)


class SampleMetricResult(BaseModel):
    dataset_name: str
    mode: str
    task_description: str
    question: str
    faithfulness: float | None
    context_relevance: float | None
    response_relevance: float | None
    errors: dict[str, str] = Field(default_factory=dict)


class SampleIntermediateResult(BaseModel):
    dataset_name: str
    mode: str
    task_description: str
    question: str
    metrics: dict[str, dict[str, Any]]


@dataclass
class MetricBundle:
    faithfulness: Any
    context_relevance: Any
    response_relevance: Any


class LocalSentenceTransformerEmbeddings:
    """Ragas-compatible embedding wrapper using local SentenceTransformer."""

    def __init__(self, model_name: str, device: str, batch_size: int):
        from ragas.embeddings.base import BaseRagasEmbeddings
        from ragas.run_config import RunConfig
        from sentence_transformers import SentenceTransformer

        class _Impl(BaseRagasEmbeddings):
            def __init__(self, inner_model_name: str, inner_device: str, inner_batch_size: int):
                super().__init__()
                self.model = SentenceTransformer(inner_model_name, device=inner_device)
                self.batch_size = inner_batch_size
                self.set_run_config(RunConfig())

            def embed_query(self, text: str) -> list[float]:
                return self.embed_documents([text])[0]

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                vectors = self.model.encode(
                    texts,
                    normalize_embeddings=True,
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                )
                return vectors.tolist()

            async def aembed_query(self, text: str) -> list[float]:
                return self.embed_query(text)

            async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
                return self.embed_documents(texts)

        self.impl = _Impl(model_name, device, batch_size)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.impl, name)


def run_folder(dataset_name: str) -> Path:
    return RESULTS_ROOT / f"{dataset_name}_{EXPERIMENT_NAME}"


def rag_results_file(dataset_name: str) -> Path:
    return run_folder(dataset_name) / INPUT_RAG_RESULTS_FILE_NAME


def ragas_details_file(dataset_name: str) -> Path:
    return run_folder(dataset_name) / OUTPUT_RAGAS_DETAILS_FILE_NAME


def ragas_summary_file(dataset_name: str) -> Path:
    return run_folder(dataset_name) / OUTPUT_RAGAS_SUMMARY_FILE_NAME


def ragas_intermediate_file(dataset_name: str) -> Path:
    return run_folder(dataset_name) / OUTPUT_RAGAS_INTERMEDIATE_FILE_NAME


def load_rag_rows(dataset_name: str) -> list[dict[str, Any]]:
    file_path = rag_results_file(dataset_name)
    if not file_path.exists():
        raise FileNotFoundError(f"RAG results file not found: {file_path}")

    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {file_path}")

    rows = [row for row in payload if isinstance(row, dict)]
    if not rows:
        raise ValueError(f"No valid rows found in {file_path}")

    return rows


def extract_answer(row: dict[str, Any]) -> str:
    structured = row.get("structured_answer", {})
    if isinstance(structured, dict):
        answer = str(structured.get("answer", "")).strip()
        if answer:
            return answer

    return str(row.get("raw_result", "")).strip()


def extract_retrieved_contexts(row: dict[str, Any]) -> list[str]:
    retrieved_context = row.get("retrieved_context", {})
    if not isinstance(retrieved_context, dict):
        return []

    chunks = retrieved_context.get("chunks", [])
    if not isinstance(chunks, list):
        return []

    contexts: list[str] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        content = str(chunk.get("content", "")).strip()
        if content:
            contexts.append(content)
    return contexts


def build_samples_for_mode(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []

    for row in rows:
        question = str(row.get("question", "")).strip()
        answer = extract_answer(row)
        retrieved_contexts = extract_retrieved_contexts(row)

        if not question or not answer:
            continue

        samples.append(
            {
                "user_input": question,
                "response": answer,
                "retrieved_contexts": retrieved_contexts,
                "mode": str(row.get("mode", "")).strip(),
                "dataset_name": str(row.get("dataset_name", "")).strip(),
                "task_description": str(row.get("task_description", "")).strip(),
                "question": question,
            }
        )

    return samples


def _sanitize_score(value: float | int | None) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return numeric


def _safe_json(value: Any) -> Any:
    if value is None:
        return None
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        return json.loads(json.dumps(value, default=str, ensure_ascii=False))


def _metric_result_to_trace(result: Any) -> MetricTrace:
    score = _sanitize_score(getattr(result, "value", None))
    reason = getattr(result, "reason", None)
    traces = _safe_json(getattr(result, "traces", None))

    intermediate: dict[str, Any] = {
        "result": _safe_json(result.to_dict()) if hasattr(result, "to_dict") else None,
        "traces": traces,
    }

    return MetricTrace(
        score=score,
        reason=str(reason) if reason is not None else None,
        intermediate=intermediate,
    )


def _run_with_timeout(seconds: int, fn: Any, *args: Any, **kwargs: Any) -> MetricTrace:
    timeout_seconds = max(1, int(seconds))

    def _timeout_handler(_signum: int, _frame: Any) -> None:
        raise MetricTimeoutError(f"Metric timed out after {timeout_seconds}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)
        return fn(*args, **kwargs)
    except MetricTimeoutError as exc:
        return MetricTrace(score=None, error=str(exc))
    except Exception as exc:  # noqa: BLE001
        return MetricTrace(score=None, error=str(exc))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def _build_ragas_metrics(strictness: int, request_timeout_seconds: int) -> MetricBundle:
    # Import here so the script fails with a clear message only when executed.
    from ragas.llms import llm_factory

    # Ragas has two official metric API layouts depending on version.
    try:
        from ragas.metrics.collections import (  # type: ignore[attr-defined]
            AnswerRelevancy as ResponseRelevanceMetric,
            ContextRelevance,
            Faithfulness,
        )
    except ModuleNotFoundError:
        from ragas.metrics import ContextRelevance, Faithfulness  # type: ignore[attr-defined]

        # Legacy naming in older versions uses ResponseRelevancy.
        try:
            from ragas.metrics import ResponseRelevancy as ResponseRelevanceMetric  # type: ignore[attr-defined]
        except ImportError:
            from ragas.metrics import AnswerRelevancy as ResponseRelevanceMetric  # type: ignore[attr-defined]

    # Older ragas llm_factory relies on OPENAI_* env vars.
    os.environ.setdefault("OPENAI_API_KEY", OLLAMA_API_KEY)
    os.environ.setdefault("OPENAI_BASE_URL", f"{OLLAMA_HOST}/v1")
    os.environ.setdefault("OPENAI_API_BASE", f"{OLLAMA_HOST}/v1")

    llm: Any
    try:
        client = AsyncOpenAI(
            base_url=f"{OLLAMA_HOST}/v1",
            api_key=OLLAMA_API_KEY,
            timeout=request_timeout_seconds,
        )
        llm = llm_factory(EVAL_MODEL, client=client)
    except TypeError:
        llm = llm_factory(EVAL_MODEL, base_url=f"{OLLAMA_HOST}/v1")

    embeddings = LocalSentenceTransformerEmbeddings(
        model_name=EVAL_EMBEDDING_MODEL,
        device=EVAL_EMBEDDING_DEVICE,
        batch_size=EVAL_EMBED_BATCH_SIZE,
    )

    faithfulness = Faithfulness(llm=llm)
    context_relevance = ContextRelevance(llm=llm)

    # Keep strictness configurable while remaining within the official metric implementation.
    try:
        response_relevance = ResponseRelevanceMetric(
            llm=llm,
            embeddings=embeddings,
            strictness=strictness,
        )
    except TypeError:
        response_relevance = ResponseRelevanceMetric(llm=llm, embeddings=embeddings)

    return MetricBundle(
        faithfulness=faithfulness,
        context_relevance=context_relevance,
        response_relevance=response_relevance,
    )


def _score_ragas_metric(metric: Any, **kwargs: Any) -> MetricTrace:
    if hasattr(metric, "single_turn_score"):
        from ragas.dataset_schema import SingleTurnSample

        sample = SingleTurnSample(**kwargs)
        result = metric.single_turn_score(sample)
        if isinstance(result, (int, float)):
            return MetricTrace(score=_sanitize_score(result), intermediate={"raw_result": result})
        return _metric_result_to_trace(result)

    if hasattr(metric, "score"):
        result = metric.score(**kwargs)
        if isinstance(result, (int, float)):
            return MetricTrace(score=_sanitize_score(result), intermediate={"raw_result": result})
        return _metric_result_to_trace(result)

    raise AttributeError(
        f"Unsupported metric API for {type(metric).__name__}: expected score() or single_turn_score()"
    )


def _score_faithfulness(metric: Any, sample: dict[str, Any]) -> MetricTrace:
    return _score_ragas_metric(
        metric,
        user_input=sample["user_input"],
        response=sample["response"],
        retrieved_contexts=sample["retrieved_contexts"],
    )


def _score_context_relevance(metric: Any, sample: dict[str, Any]) -> MetricTrace:
    return _score_ragas_metric(
        metric,
        user_input=sample["user_input"],
        retrieved_contexts=sample["retrieved_contexts"],
    )


def _score_response_relevance(metric: Any, sample: dict[str, Any]) -> MetricTrace:
    return _score_ragas_metric(
        metric,
        user_input=sample["user_input"],
        response=sample["response"],
    )


def evaluate_sample(
    metrics: MetricBundle,
    sample: dict[str, Any],
    metric_timeout_seconds: int,
) -> tuple[SampleMetricResult, SampleIntermediateResult]:
    faith = _run_with_timeout(metric_timeout_seconds, _score_faithfulness, metrics.faithfulness, sample)
    crel = _run_with_timeout(
        metric_timeout_seconds,
        _score_context_relevance,
        metrics.context_relevance,
        sample,
    )
    arel = _run_with_timeout(
        metric_timeout_seconds,
        _score_response_relevance,
        metrics.response_relevance,
        sample,
    )

    errors: dict[str, str] = {}
    if faith.error:
        errors["faithfulness"] = faith.error
    if crel.error:
        errors["context_relevance"] = crel.error
    if arel.error:
        errors["response_relevance"] = arel.error

    sample_result = SampleMetricResult(
        dataset_name=sample["dataset_name"],
        mode=sample["mode"],
        task_description=sample["task_description"],
        question=sample["question"],
        faithfulness=_sanitize_score(faith.score),
        context_relevance=_sanitize_score(crel.score),
        response_relevance=_sanitize_score(arel.score),
        errors=errors,
    )

    intermediate_result = SampleIntermediateResult(
        dataset_name=sample["dataset_name"],
        mode=sample["mode"],
        task_description=sample["task_description"],
        question=sample["question"],
        metrics={
            "faithfulness": {
                "score": _sanitize_score(faith.score),
                "reason": faith.reason,
                "error": faith.error,
                "intermediate": faith.intermediate,
            },
            "context_relevance": {
                "score": _sanitize_score(crel.score),
                "reason": crel.reason,
                "error": crel.error,
                "intermediate": crel.intermediate,
            },
            "response_relevance": {
                "score": _sanitize_score(arel.score),
                "reason": arel.reason,
                "error": arel.error,
                "intermediate": arel.intermediate,
            },
        },
    )

    return sample_result, intermediate_result


def summarize_details(details: list[dict[str, Any]]) -> dict[str, float | None]:
    def safe_avg(key: str) -> float | None:
        values: list[float] = []
        for row in details:
            value = row.get(key)
            if isinstance(value, (int, float)):
                numeric = float(value)
                if math.isfinite(numeric):
                    values.append(numeric)
        if not values:
            return None
        avg = sum(values) / len(values)
        return avg if math.isfinite(avg) else None

    return {
        "faithfulness_avg": safe_avg("faithfulness"),
        "context_relevance_avg": safe_avg("context_relevance"),
        "response_relevance_avg": safe_avg("response_relevance"),
    }


def evaluate_dataset(
    dataset_name: str,
    modes: set[str] | None,
    max_samples_per_mode: int | None,
    save_every_samples: int,
    strictness: int,
    request_timeout_seconds: int,
) -> None:
    rows = load_rag_rows(dataset_name)

    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        mode = str(row.get("mode", "")).strip() or "unknown"
        by_mode.setdefault(mode, []).append(row)

    details_payload: dict[str, list[dict[str, Any]]] = {}
    intermediate_payload: dict[str, list[dict[str, Any]]] = {}
    summary_payload: dict[str, Any] = {
        "dataset_name": dataset_name,
        "modes": {},
    }

    details_file = ragas_details_file(dataset_name)
    summary_file = ragas_summary_file(dataset_name)
    intermediate_file = ragas_intermediate_file(dataset_name)

    def save_checkpoint() -> None:
        details_file.write_text(
            json.dumps(details_payload, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        summary_file.write_text(
            json.dumps(summary_payload, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        intermediate_file.write_text(
            json.dumps(intermediate_payload, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )

    save_checkpoint()

    metrics = _build_ragas_metrics(
        strictness=strictness,
        request_timeout_seconds=request_timeout_seconds,
    )
    metric_timeout_seconds = request_timeout_seconds + METRIC_TIMEOUT_BUFFER_SECONDS

    for mode, mode_rows in by_mode.items():
        normalized_mode = mode.strip().lower()
        if modes is not None and normalized_mode not in modes:
            continue

        print(f"[{dataset_name}] computing official Ragas metrics for mode={mode} on {len(mode_rows)} rows")
        samples = build_samples_for_mode(mode_rows)
        if max_samples_per_mode is not None:
            samples = samples[:max_samples_per_mode]
            print(f"[{dataset_name}] mode={mode} truncated to {len(samples)} samples")

        mode_details: list[dict[str, Any]] = []
        mode_intermediates: list[dict[str, Any]] = []
        for idx, sample in enumerate(samples, start=1):
            print(f"[{dataset_name}] mode={mode} evaluating sample {idx}/{len(samples)}")
            try:
                result, intermediate = evaluate_sample(
                    metrics,
                    sample,
                    metric_timeout_seconds=metric_timeout_seconds,
                )
                mode_details.append(result.model_dump())
                mode_intermediates.append(intermediate.model_dump())
            except KeyboardInterrupt:
                details_payload[mode] = mode_details
                intermediate_payload[mode] = mode_intermediates
                summary_payload["modes"][mode] = {
                    "count": len(mode_details),
                    **summarize_details(mode_details),
                }
                save_checkpoint()
                raise
            except Exception as exc:  # noqa: BLE001
                fallback = SampleMetricResult(
                    dataset_name=sample["dataset_name"],
                    mode=sample["mode"],
                    task_description=sample["task_description"],
                    question=sample["question"],
                    faithfulness=None,
                    context_relevance=None,
                    response_relevance=None,
                    errors={"fatal": str(exc)},
                )
                mode_details.append(fallback.model_dump())
                mode_intermediates.append(
                    SampleIntermediateResult(
                        dataset_name=sample["dataset_name"],
                        mode=sample["mode"],
                        task_description=sample["task_description"],
                        question=sample["question"],
                        metrics={
                            "faithfulness": {
                                "score": None,
                                "error": "Not computed due to fatal error",
                                "intermediate": {},
                            },
                            "context_relevance": {
                                "score": None,
                                "error": "Not computed due to fatal error",
                                "intermediate": {},
                            },
                            "response_relevance": {
                                "score": None,
                                "error": "Not computed due to fatal error",
                                "intermediate": {},
                            },
                            "fatal": {"error": str(exc)},
                        },
                    ).model_dump()
                )

            if idx % max(1, save_every_samples) == 0 or idx == len(samples):
                details_payload[mode] = mode_details
                intermediate_payload[mode] = mode_intermediates
                summary_payload["modes"][mode] = {
                    "count": len(mode_details),
                    **summarize_details(mode_details),
                }
                save_checkpoint()

        if mode not in details_payload:
            details_payload[mode] = []
            intermediate_payload[mode] = []
            summary_payload["modes"][mode] = {
                "count": 0,
                "faithfulness_avg": None,
                "context_relevance_avg": None,
                "response_relevance_avg": None,
            }
            save_checkpoint()

    print(f"[{dataset_name}] RAGAS details written to {details_file}")
    print(f"[{dataset_name}] RAGAS intermediate traces written to {intermediate_file}")
    print(f"[{dataset_name}] RAGAS summary written to {summary_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute official Ragas metrics and persist metric traces/intermediates"
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=DATASET_NAMES,
        help="Datasets to evaluate (default: configured DATASET_NAMES)",
    )
    parser.add_argument(
        "--modes",
        nargs="*",
        default=None,
        help="Optional mode filter (e.g. hybrid naive)",
    )
    parser.add_argument(
        "--max-samples-per-mode",
        type=int,
        default=None,
        help="Optional cap of samples per mode for faster runs",
    )
    parser.add_argument(
        "--save-every-samples",
        type=int,
        default=SAVE_EVERY_SAMPLES,
        help="Write partial outputs after each N samples",
    )
    parser.add_argument(
        "--strictness",
        type=int,
        default=STRICTNESS,
        help="Strictness for official Answer Relevancy metric",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=int,
        default=REQUEST_TIMEOUT_SECONDS,
        help="Per-request timeout for Ragas evaluator calls",
    )
    args = parser.parse_args()

    modes = {m.strip().lower() for m in args.modes} if args.modes else None
    max_samples_per_mode = (
        args.max_samples_per_mode
        if args.max_samples_per_mode is not None and args.max_samples_per_mode > 0
        else None
    )
    save_every_samples = max(1, args.save_every_samples)
    strictness = max(1, args.strictness)
    request_timeout_seconds = max(1, args.request_timeout_seconds)

    for dataset_name in args.datasets:
        evaluate_dataset(
            dataset_name,
            modes=modes,
            max_samples_per_mode=max_samples_per_mode,
            save_every_samples=save_every_samples,
            strictness=strictness,
            request_timeout_seconds=request_timeout_seconds,
        )


if __name__ == "__main__":
    main()
