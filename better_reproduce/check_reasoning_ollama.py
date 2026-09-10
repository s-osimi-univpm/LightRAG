from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

# =========================
# Config (edit these)
# =========================
OLLAMA_HOST = "http://localhost:11440"
MODEL = "gemma4:e4b-it-q4_K_M"
PROMPT = "Briefly answer: what is 37 * 29? Also include one sentence explaining your method."
TIMEOUT_SECONDS = 300
AUDIT_DIR = Path("better_reproduce/results/reasoning_audit")
RUN_THINK_TRUE = True
RUN_THINK_FALSE = True


def collect_reasoning_text(node: Any, path: str = "") -> list[tuple[str, str]]:
    hits: list[tuple[str, str]] = []
    reasoning_keys = {
        "thinking",
        "reasoning",
        "reasoning_content",
        "thought",
        "thoughts",
    }

    if isinstance(node, dict):
        for key, value in node.items():
            next_path = f"{path}.{key}" if path else key
            lowered = key.lower()
            if lowered in reasoning_keys:
                if isinstance(value, str):
                    hits.append((next_path, value))
                else:
                    hits.append((next_path, json.dumps(value, ensure_ascii=False)))
            if lowered == "content" and isinstance(value, str):
                content_lowered = value.lower()
                if "<think>" in content_lowered and "</think>" in content_lowered:
                    hits.append((next_path, value))
            hits.extend(collect_reasoning_text(value, next_path))
    elif isinstance(node, list):
        for idx, item in enumerate(node):
            next_path = f"{path}[{idx}]"
            hits.extend(collect_reasoning_text(item, next_path))

    return hits


def call_ollama_chat(host: str, model: str, prompt: str, think: bool, timeout: int) -> tuple[dict[str, Any], float]:
    url = f"{host.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": think,
    }

    start = time.perf_counter()
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    elapsed_seconds = time.perf_counter() - start
    return response.json(), elapsed_seconds


def extract_answer_and_reasoning(data: dict[str, Any]) -> tuple[str, str, str]:
    message = data.get("message", {}) if isinstance(data, dict) else {}
    content = message.get("content", "") if isinstance(message, dict) else ""
    content = content if isinstance(content, str) else ""

    reasoning_from_tags = ""
    raw_answer = content.strip()
    answer = content.strip()
    match = re.search(r"<think>(.*?)</think>", content, flags=re.DOTALL | re.IGNORECASE)
    if match:
        reasoning_from_tags = match.group(1).strip()
        answer = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE).strip()

    reasoning_hits = collect_reasoning_text(data)
    reasoning_values: list[str] = []
    for field_path, value in reasoning_hits:
        if field_path.endswith("content") and "<think>" in value.lower() and "</think>" in value.lower():
            continue
        clean = value.strip()
        if clean:
            reasoning_values.append(clean)

    merged_reasoning = reasoning_from_tags
    if not merged_reasoning and reasoning_values:
        merged_reasoning = "\n\n".join(dict.fromkeys(reasoning_values))

    return raw_answer, merged_reasoning, answer


def sanitize_filename(value: str) -> str:
    sanitized = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_", "."}:
            sanitized.append(ch)
        else:
            sanitized.append("_")
    return "".join(sanitized).strip("_") or "model"


def write_audit_file(audit_dir: Path, model: str, payload: dict[str, Any]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = sanitize_filename(model)
    audit_dir.mkdir(parents=True, exist_ok=True)
    output_file = audit_dir / f"{timestamp}_{model_slug}.json"
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_file


def main() -> None:
    print("Checking reasoning behavior with Ollama chat API")
    print(f"host={OLLAMA_HOST}")
    print(f"model={MODEL}")

    think_values: list[bool] = []
    if RUN_THINK_TRUE:
        think_values.append(True)
    if RUN_THINK_FALSE:
        think_values.append(False)
    if not think_values:
        raise ValueError("Enable at least one of RUN_THINK_TRUE or RUN_THINK_FALSE.")

    runs: list[dict[str, Any]] = []
    for think in think_values:
        response, elapsed_seconds = call_ollama_chat(
            host=OLLAMA_HOST,
            model=MODEL,
            prompt=PROMPT,
            think=think,
            timeout=TIMEOUT_SECONDS,
        )
        raw_answer, reasoning, result = extract_answer_and_reasoning(response)
        total_duration = response.get("total_duration")
        reasoning_duration = response.get("reasoning_duration")
        thinking_duration = response.get("thinking_duration")
        thinking_time_seconds = None
        reasoning_time_seconds = None
        total_time_seconds = round(elapsed_seconds, 3)

        if isinstance(total_duration, (int, float)):
            thinking_time_seconds = total_duration / 1_000_000_000
        elif think and reasoning:
            thinking_time_seconds = elapsed_seconds

        if isinstance(reasoning_duration, (int, float)):
            reasoning_time_seconds = reasoning_duration / 1_000_000_000
        elif isinstance(thinking_duration, (int, float)):
            reasoning_time_seconds = thinking_duration / 1_000_000_000
        elif think and reasoning:
            reasoning_time_seconds = thinking_time_seconds

        runs.append(
            {
                "think": think,
                "raw_answer": raw_answer,
                "thinking": reasoning,
                "result": result,
                "thinking_time_seconds": round(thinking_time_seconds, 3) if thinking_time_seconds is not None else None,
                "reasoning_time_seconds": round(reasoning_time_seconds, 3) if reasoning_time_seconds is not None else None,
                "total_time_seconds": total_time_seconds,
            }
        )
        print(
            f"think={think} | total_time_seconds={total_time_seconds} | "
            f"thinking_time_seconds={round(thinking_time_seconds, 3) if thinking_time_seconds is not None else 'n/a'} | "
            f"reasoning_time_seconds={round(reasoning_time_seconds, 3) if reasoning_time_seconds is not None else 'n/a'}"
        )

    output_payload = {
        "created_at": datetime.now().isoformat(),
        "host": OLLAMA_HOST,
        "model": MODEL,
        "prompt": PROMPT,
        "timeout_seconds": TIMEOUT_SECONDS,
        "runs": runs,
    }
    output_file = write_audit_file(AUDIT_DIR, MODEL, output_payload)
    print(f"\nOutput written to: {output_file}")


if __name__ == "__main__":
    main()
