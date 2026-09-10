# better_reproduce: Script-by-Script Documentation

This folder contains a multi-step benchmark pipeline around LightRAG:

1. Build LightRAG indexes from dataset context files.
2. Generate user personas, tasks, and grounded questions.
3. Run RAG answers in multiple retrieval modes.
4. Evaluate answer quality between modes.
5. Audit whether Ollama reasoning output is disabled when requested.

The scripts are designed to run independently, but they are intended to be used in order.

## Folder Overview

Top-level Python files documented here:

- `1_indexing.py`
- `2_questions.py`
- `3_RAG.py`
- `eval_rag_quality_from_3_rag.py`
- `batch_eval_tie.py`
- `batch_eval.py`
- `check_reasoning_ollama.py`
- `functions.py`

Main data/result roots used by these scripts:

- Input dataset contexts: `better_reproduce/dataset/unique_contexts/`
- Step 1 outputs (indexing): `better_reproduce/results/1_indexing/`
- Step 2 outputs (questions): `better_reproduce/results/2_questions/`
- Step 3 outputs (RAG + eval): `better_reproduce/results/3_rag/`

---

## End-to-End Pipeline

Typical execution order:

1. `python better_reproduce/1_indexing.py`
2. `python better_reproduce/2_questions.py`
3. `python better_reproduce/3_RAG.py`
4. `python better_reproduce/batch_eval_tie.py` (or `--no-tie`)
5. Optional: `python better_reproduce/batch_eval.py`
6. Optional: `python better_reproduce/check_reasoning_ollama.py --save-raw`

Evaluate precomputed 3_RAG outputs directly with RAGAS (without calling API again):

`python better_reproduce/eval_rag_quality_from_3_rag.py --rag-results better_reproduce/results/3_rag/<run_folder>/rag_results.json --mode hybrid`

Each step writes its own constants snapshot (`constants.txt`) into that step's output directory for reproducibility.

---

## 1) `1_indexing.py`

### Purpose
Builds LightRAG stores from dataset context JSON files. This is the ingestion and indexing stage.

### Inputs
- Per dataset input file:
  - `better_reproduce/dataset/unique_contexts/{dataset}_unique_contexts.json`
- Default datasets:
  - `agriculture`, `legal`, `mix`

Expected input shape: a JSON list of strings (unique context passages).

### Core Behavior
- Loads a SentenceTransformer embedding model (`BAAI/bge-m3`) on configured device.
- Creates a LightRAG instance using Ollama (`gemma4:31b` by default).
- Joins all contexts into one text blob separated by blank lines.
- Enqueues and processes documents through LightRAG pipeline.
- Supports configurable chunking strategy:
  - Shorthand: `F`, `R`, `V`, `P`
  - Aliases: `fixed_token`, `recursive_character`, `semantic_vector`, `paragraph_semantic`

### Important Constants
- `EXPERIMENT_NAME`: output folder suffix (e.g. `gemma31b_no_reason`)
- `CHUNKING_STRATEGY`: ingestion chunking mode
- `OVERWRITE_EXISTING_INDEX`: whether to delete existing index files first
- `ENABLE_REASONING`: passed to Ollama `think` parameter

### Outputs
Per dataset output directory:
- `better_reproduce/results/1_indexing/{dataset}_{EXPERIMENT_NAME}/`

Generated files include LightRAG stores such as:
- `kv_store_text_chunks.json`
- `kv_store_full_docs.json`
- `vdb_chunks.json`
- `vdb_entities.json`
- `vdb_relationships.json`
- `graph_chunk_entity_relation.graphml`
- plus logs and constants snapshot:
  - `step_1.log`
  - `constants.txt`

### Failure Handling
- Missing input file raises `FileNotFoundError`.
- Controlled by `CONTINUE_ON_DATASET_ERROR` to continue/stop across datasets.

---

## 2) `2_questions.py`

### Purpose
Generates synthetic benchmark queries from indexed chunks:
- 1 persona per generation
- multiple tasks per persona
- multiple naive user questions per task
- per-question supporting chunk IDs (top evidence chunks)

### Inputs
- Step 1 chunk store:
  - `better_reproduce/results/1_indexing/{dataset}_{STEP_1_EXPERIMENT_NAME}/kv_store_text_chunks.json`

### Core Behavior
1. Loads chunk store and filters short chunks (`MIN_CHUNK_CHARS`).
2. Shuffles chunks deterministically by seed.
3. Builds non-overlapping per-user chunk pools.
4. Samples up to `CHUNKS_PER_DATASET` chunks per user.
5. Builds a textual description from sampled chunks and prompts an LLM via `instructor` structured output.
6. Enforces generation constraints with retries:
   - exact task count
   - exact question count per task
   - valid per-question supporting chunk IDs

### Prompt/Style Constraints
The current prompt explicitly requires:
- Questions are naive end-user style (no internal chunk awareness).
- No references to chunk IDs/sections/file paths/context scaffolding.
- No direct source citation language.
- Every question must include `top_supporting_chunk_ids` with 1 to 3 IDs from an allowed sampled set.

### Output Schema Highlights
Each generated question includes:
- `question` (string)
- `top_supporting_chunk_ids` (list of chunk IDs)

Top-level output includes per user:
- `user_description`
- `query` marker
- `sampled_chunks` (full sampled chunk metadata/content)
- `task_questions`

### Outputs
Per dataset output directory:
- `better_reproduce/results/2_questions/{dataset}_{EXPERIMENT_NAME}/`

Files:
- `questions.json`: full structured generation output
- `chunk_samples.json`: sampling metadata and generation mode
- `step_2.log`
- `constants.txt`

### Failure Handling
- Raises if chunk file missing or no usable chunks.
- Retries generation up to `MAX_GENERATION_RETRIES` when shape/evidence constraints fail.

---

## 3) `3_RAG.py`

### Purpose
Runs all generated questions through LightRAG query modes and records:
- model answers
- retrieved context
- retrieval-hit metrics against sampled chunk IDs
- run errors

### Inputs
- Step 1 indexing folder mapping per dataset (`INDEXING_RESULT_FOLDER_BY_DATASET`)
- Step 2 questions file mapping per dataset (`QUESTIONS_RESULT_FOLDER_BY_DATASET`)
- Questions file: `questions.json`

### Core Behavior
1. Loads questions into `QueryWorkItem` list.
2. Initializes LightRAG with selected index folder.
3. For each query mode (default: `hybrid`, `naive`):
   - Sends a structured prompt requiring JSON answer:
     - `answer`
     - `confidence` in [0,1]
     - `assumptions`
   - Extracts retrieved context (`entities`, `relationships`, `chunks`).
   - Computes retrieval metrics:
     - sampled/retrieved chunk counts
     - hit count and hit rate
     - graph-linked hit count (hybrid)
     - entity contribution index (hybrid)
     - estimated retrieved-context token count
4. Appends each result immediately into JSON array files (incremental persistence).

### Environment Overrides
Runtime parameters can be overridden with env vars:
- `RAG_NUM_CTX`
- `RAG_REQUEST_TIMEOUT_SECONDS`
- `EMBEDDING_DIM`
- `MAX_EMBED_TOKENS`

### Outputs
Per dataset output directory:
- `better_reproduce/results/3_rag/{dataset}_{EXPERIMENT_NAME}/`

Files:
- `rag_results.json`: all successful query results across modes
- `rag_errors.json`: per-query errors
- `rag_run_summary.json`: summary object with counts and folder mappings
- `3_rag.log`
- `constants.txt`

### Notes
- `rag_results.json` is a flat list where each row includes a `mode` field.
- The evaluator scripts assume rows are pairable by question between `hybrid` and `naive`.

---

## 4) `batch_eval_tie.py`

### Purpose
Compares two RAG modes answer-by-answer using an LLM judge and produces:
- detailed per-question evaluation
- aggregate summary stats

By default compares:
- `MODE_1 = hybrid`
- `MODE_2 = naive`

### Inputs
- `better_reproduce/results/3_rag/{dataset}_{EXPERIMENT_NAME}/rag_results.json`

### Core Behavior
1. Splits rows by mode and pairs them by index.
2. Verifies paired questions are identical.
3. Sends both answers to evaluator LLM (`gemma4:31b`) using structured output.
4. Scores criteria:
   - Comprehensiveness
   - Diversity
   - Empowerment
   - Overall Winner
5. Also computes:
   - tie decision metadata (if ties enabled)
   - deterministic final winner (`Answer 1` or `Answer 2`)
6. Aggregates summary with:
   - per-criterion winner counts
   - final winner counts
   - no-context signal counts
   - retrieval metric averages by mode

### Tie Behavior
- Default: ties allowed (`ALLOW_TIES_BY_DEFAULT = True`).
- CLI option `--no-tie` disables tie labels and writes `_no_tie` outputs.

### Outputs
Per dataset run folder:
- Tie-enabled mode:
  - `evaluation.json`
  - `summary.json`
- No-tie mode (`--no-tie`):
  - `evaluation_no_tie.json`
  - `summary_no_tie.json`

---

## 5) `batch_eval.py`

### Purpose
Dedicated no-tie evaluator with resumable progress and stricter forced winner output.

### Inputs
- Same as `batch_eval_tie.py`: `rag_results.json`

### Core Behavior
- Imports shared components from `batch_eval_tie.py`.
- Uses a no-tie response schema (`Answer 1` or `Answer 2` only).
- Supports incremental resume from existing evaluation file:
  - `RESUME_FROM_EXISTING_EVAL = True`
- After every evaluated question, writes progress to disk:
  - evaluation rows
  - updated summary
- Handles and retries on:
  - event loop binding errors
  - incomplete output exceptions

### Outputs
Per dataset run folder:
- `evaluation_no_tie.json`
- `summary_no_tie.json`

---

## 6) `check_reasoning_ollama.py`

### Purpose
Audits whether Ollama exposes reasoning/thinking fields when `think=True`, and whether those disappear when `think=False`.

Useful when validating model serving behavior for reproducibility.

### Inputs
None from pipeline artifacts; this script directly calls Ollama `/api/chat`.

### Core Behavior
1. Calls Ollama twice with same prompt:
   - `think=True`
   - `think=False`
2. Recursively scans JSON responses for reasoning-related fields such as:
   - `thinking`, `reasoning`, `reasoning_content`, `thought`, `thoughts`
   - or `<think>...</think>` tags in content
3. Prints side-by-side summary and verdict.
4. Optionally saves raw JSON payloads.

### CLI Options
- `--host` (default `http://localhost:11440`)
- `--model`
- `--prompt`
- `--timeout`
- `--save-raw`
- `--audit-dir`

### Outputs
- Console verdict (`PASS`, `FAIL`, or `INCONCLUSIVE`).
- Optional saved files (with `--save-raw`):
  - `{audit_dir}/{timestamp}_{model_slug}/response_think_true.json`
  - `{audit_dir}/{timestamp}_{model_slug}/response_think_false.json`

---

## 7) `functions.py`

### Purpose
Shared helpers used by step scripts.

### Functions
- `initialize_logger(log_dir, log_filename, verbose_debug)`
  - Configures LightRAG logging (console + rotating file handler).
  - Clears pre-existing handlers for selected logger names.
  - Returns resolved log file path.

- `write_constants_snapshot(output_dir, constants, file_name="constants.txt")`
  - Writes all uppercase, non-callable constants into a text file.
  - Converts `Path` values to strings.
  - Used for experiment reproducibility and traceability.

---

## Practical Notes

- Make sure your Ollama endpoint and model names in each script match what is actually served.
- Some defaults differ across scripts (for example, dataset lists and experiment names), so verify constants before running.
- `2_questions.py` and `3_RAG.py` depend on consistent folder naming between steps.
- If you rerun with changed constants, prefer a new `EXPERIMENT_NAME` to avoid mixing artifacts.

---

## Quick Run Commands

From repo root:

```bash
python better_reproduce/1_indexing.py
python better_reproduce/2_questions.py
python better_reproduce/3_RAG.py
python better_reproduce/batch_eval_tie.py
python better_reproduce/batch_eval_tie.py --no-tie
python better_reproduce/check_reasoning_ollama.py --save-raw
```

If running from inside `better_reproduce/`, adjust paths accordingly.
