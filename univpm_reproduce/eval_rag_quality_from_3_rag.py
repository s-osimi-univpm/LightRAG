#!/usr/bin/env python3
"""Evaluate precomputed 3_RAG results with RAGAS metrics.

This script reads entries from:
  univpm_reproduce/results/3_rag/<run_folder>/rag_results.json

and evaluates each (question, answer, contexts, ground_truth) tuple directly,
without calling LightRAG API again.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

warnings.filterwarnings(
    "ignore",
    message=".*LangchainLLMWrapper is deprecated.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*Unexpected type for token usage.*",
    category=UserWarning,
)

try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except ImportError as exc:
    raise ImportError(
        "Missing dependencies. Install with: pip install ragas datasets langchain-openai"
    ) from exc


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _clean_score(value: Any) -> Any:
    if _is_nan(value):
        return None
    return value


def _extract_answer(entry: dict[str, Any]) -> str:
    structured = entry.get("structured_answer")
    if isinstance(structured, dict):
        answer = str(structured.get("answer", "")).strip()
        if answer:
            return answer
    return str(entry.get("raw_result", "")).strip()


def _extract_contexts(entry: dict[str, Any]) -> list[str]:
    retrieved_context = entry.get("retrieved_context")
    if not isinstance(retrieved_context, dict):
        return []

    contexts: list[str] = []

    chunks = retrieved_context.get("chunks", [])
    if isinstance(chunks, list):
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            content = str(chunk.get("content", "")).strip()
            if content:
                contexts.append(content)

    if contexts:
        return contexts

    entities = retrieved_context.get("entities", [])
    if isinstance(entities, list):
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            description = str(entity.get("description", "")).strip()
            if description:
                contexts.append(description)

    return contexts


def _load_eval_items(rag_results_path: Path, mode: str) -> list[dict[str, Any]]:
    payload = json.loads(rag_results_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {rag_results_path}")

    items: list[dict[str, Any]] = []
    for idx, entry in enumerate(payload, start=1):
        if not isinstance(entry, dict):
            continue

        if mode != "all" and entry.get("mode") != mode:
            continue

        question = str(entry.get("question", "")).strip()
        ground_truth = str(entry.get("reference_answer", "")).strip()
        answer = _extract_answer(entry)
        contexts = _extract_contexts(entry)

        if not question or not answer or not ground_truth:
            continue

        items.append(
            {
                "idx": idx,
                "mode": str(entry.get("mode", "")),
                "dataset_name": str(entry.get("dataset_name", "")),
                "question": question,
                "answer": answer,
                "ground_truth": ground_truth,
                "contexts": contexts,
            }
        )
    return items


def _build_eval_models() -> tuple[Any, Any]:
    eval_llm_api_key = os.getenv("EVAL_LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not eval_llm_api_key:
        raise EnvironmentError(
            "EVAL_LLM_BINDING_API_KEY or OPENAI_API_KEY is required for evaluation"
        )

    eval_model = os.getenv("EVAL_LLM_MODEL", "gpt-4o-mini")
    eval_llm_base_url = os.getenv("EVAL_LLM_BINDING_HOST")

    eval_embedding_api_key = (
        os.getenv("EVAL_EMBEDDING_BINDING_API_KEY")
        or os.getenv("EVAL_LLM_BINDING_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    eval_embedding_model = os.getenv("EVAL_EMBEDDING_MODEL", "text-embedding-3-large")
    eval_embedding_base_url = os.getenv("EVAL_EMBEDDING_BINDING_HOST") or os.getenv(
        "EVAL_LLM_BINDING_HOST"
    )

    llm_kwargs = {
        "model": eval_model,
        "api_key": eval_llm_api_key,
        "max_retries": int(os.getenv("EVAL_LLM_MAX_RETRIES", "5")),
        "request_timeout": int(os.getenv("EVAL_LLM_TIMEOUT", "180")),
    }
    embedding_kwargs = {
        "model": eval_embedding_model,
        "api_key": eval_embedding_api_key,
    }

    if eval_llm_base_url:
        llm_kwargs["base_url"] = eval_llm_base_url
    if eval_embedding_base_url:
        embedding_kwargs["base_url"] = eval_embedding_base_url

    base_llm = ChatOpenAI(**llm_kwargs)
    embeddings = OpenAIEmbeddings(**embedding_kwargs)

    try:
        llm = LangchainLLMWrapper(langchain_llm=base_llm, bypass_n=True)
    except Exception:
        llm = base_llm

    return llm, embeddings


def _save_results(
    items: list[dict[str, Any]],
    scores_df: Any,
    output_dir: Path,
    mode: str,
) -> tuple[Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"ragas_from_3_rag_{mode}_{timestamp}.json"
    csv_path = output_dir / f"ragas_from_3_rag_{mode}_{timestamp}.csv"

    merged_results: list[dict[str, Any]] = []
    for item, (_, score_row) in zip(items, scores_df.iterrows()):
        metrics = {
            "faithfulness": _clean_score(score_row.get("faithfulness")),
            "answer_relevancy": _clean_score(score_row.get("answer_relevancy")),
            "context_recall": _clean_score(score_row.get("context_recall")),
            "context_precision": _clean_score(score_row.get("context_precision")),
        }
        merged_results.append(
            {
                **item,
                "context_count": len(item["contexts"]),
                "metrics": metrics,
            }
        )

    summary = {
        "mode": mode,
        "total": len(merged_results),
        "avg_faithfulness": _clean_score(scores_df["faithfulness"].mean()),
        "avg_answer_relevancy": _clean_score(scores_df["answer_relevancy"].mean()),
        "avg_context_recall": _clean_score(scores_df["context_recall"].mean()),
        "avg_context_precision": _clean_score(scores_df["context_precision"].mean()),
    }

    json_path.write_text(
        json.dumps({"summary": summary, "results": merged_results}, indent=2),
        encoding="utf-8",
    )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "idx",
                "dataset_name",
                "mode",
                "question",
                "context_count",
                "faithfulness",
                "answer_relevancy",
                "context_recall",
                "context_precision",
            ]
        )
        for row in merged_results:
            writer.writerow(
                [
                    row["idx"],
                    row["dataset_name"],
                    row["mode"],
                    row["question"],
                    row["context_count"],
                    row["metrics"].get("faithfulness"),
                    row["metrics"].get("answer_relevancy"),
                    row["metrics"].get("context_recall"),
                    row["metrics"].get("context_precision"),
                ]
            )

    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate precomputed 3_RAG answers with RAGAS",
    )
    parser.add_argument(
        "--rag-results",
        "-i",
        type=str,
        required=True,
        help="Path to rag_results.json",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["all", "hybrid", "naive"],
        default="all",
        help="Filter entries by mode before evaluation",
    )

    args = parser.parse_args()

    load_dotenv(dotenv_path=".env", override=False)

    rag_results_path = Path(args.rag_results).resolve()
    if not rag_results_path.exists():
        raise FileNotFoundError(f"rag_results.json not found: {rag_results_path}")

    items = _load_eval_items(rag_results_path, args.mode)
    if not items:
        raise ValueError("No valid items found after filtering. Check input path/mode.")

    dataset = Dataset.from_dict(
        {
            "question": [item["question"] for item in items],
            "answer": [item["answer"] for item in items],
            "contexts": [item["contexts"] for item in items],
            "ground_truth": [item["ground_truth"] for item in items],
        }
    )

    llm, embeddings = _build_eval_models()

    eval_results = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextRecall(), ContextPrecision()],
        llm=llm,
        embeddings=embeddings,
    )
    scores_df = eval_results.to_pandas()

    output_dir = rag_results_path.parent
    json_path, csv_path = _save_results(items, scores_df, output_dir, args.mode)

    print(f"Evaluated {len(items)} entries")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
