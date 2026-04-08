---
title: OpenEnv PaperBench
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: apache-2.0
app_port: 7860
---


# OpenEnv-PaperBench

OpenEnv-PaperBench is a sequential, budget-constrained research paper screening environment for agent evaluation.

An agent acts like a program committee member or systematic-review screener. It reads a batch of synthetic paper abstracts, makes screening decisions under a fixed review budget, and is scored by deterministic graders.

It is a structured decision-making task based on a real workflow people already do.

## What this project contains

- 4 progressively harder tasks
- Typed Pydantic models for actions and observations
- A deterministic OpenEnv-compatible server
- A baseline `inference.py`
- Docker deployment for Hugging Face Spaces
- Deterministic, rule-based graders
- Full local validation support

## Task overview

### Task 1 — Binary relevance screening
Label each paper as:

- `RELEVANT`
- `NOT_RELEVANT`

This is the easiest task. The agent only needs to decide whether the paper matches the topic.

### Task 2 — Relevance + quality scoring
Each paper gets:

- a relevance label
- a quality score from `1` to `4`

This task checks whether the agent can judge both topic fit and methodological strength.

### Task 3 — Adversarial batch screening
Each paper is labeled as:

- `INCLUDE`
- `EXCLUDE`
- `DEFER`

This task introduces red-herring papers and budget pressure.

### Task 4 — Ranking and justification under budget pressure
The agent must produce a ranked top-5 shortlist.

`INCLUDE` decisions require:

- `rank` from `1` to `5`
- a short justification

This is the hardest task and the closest to a real screening workflow.

## Environment design

The environment exposes the standard OpenEnv-style flow:

- `reset()`
- `step()`
- `state()`

Each episode has:

- a fixed task
- a fixed paper batch
- a fixed step budget
- deterministic scoring

The environment is API-first. It is meant to be driven by code, not by a browser UI.

## API endpoints

### `GET /health`
Returns a simple liveness check.

Expected response:

```json
{"status":"ok","env":"paper_review_env_v1"}
```

### `GET /tasks`

Lists the available tasks.

### `POST /reset`

Starts a new episode.

Required body:

```json
{
  "task_id": "task1",
  "instance_id": "instance_001"
}
```

### `POST /step`

Advances the episode by one action.

Required body:

```json
{
  "session_id": "...",
  "action": { ... }
}
```

## Action schema

The environment uses one unified action schema.

Supported fields include:

* `action_type`
* `paper_id`
* `label`
* `quality_score`
* `rank`
* `justification`

The exact validation rules depend on the active task.

## Observation schema

Each observation includes:

* `task_id`
* `task_description`
* `step`
* `budget_remaining`
* `papers`
* `decisions_so_far`
* `episode_complete`
* `final_score`
* `error`

The observation is intentionally flat and easy to serialize.

## Scoring

All scores are deterministic and normalized to `[0.0, 1.0]`.

### Task 1

Binary F1 score on relevance labels.

### Task 2

`0.6 × relevance F1 + 0.4 × quality accuracy`

### Task 3

`0.85 × F1 + 0.15 × budget efficiency`

### Task 4

`0.50 × nDCG@5 + 0.35 × F1 + 0.15 × justification validity`

## Local baseline scores

These are the current local baseline results from `inference.py` on the included fixtures.

| Task   | Score |
| ------ | ----: |
| Task 1 |  0.67 |
| Task 2 |  0.68 |
| Task 3 |  0.54 |
| Task 4 |  0.87 |
| Mean   |  0.68 |

These numbers are useful for local verification.

## Project structure

```text
openenv-paperbench/
├── Dockerfile
├── README.md
├── openenv.yaml
├── pyproject.toml
├── uv.lock
├── inference.py
├── env/
├── tasks/
├── graders/
├── fixtures/
└── server/
```

## Requirements

* Python 3.10+
* Docker
* Hugging Face account for Space deployment
* OpenAI-compatible API access for the optional LLM baseline path

## Environment variables

### For `inference.py`

* `API_BASE_URL`
* `MODEL_NAME`
* `HF_TOKEN`
* `ENV_BASE_URL`
* `BENCHMARK`
* `TASK_IDS`
* `INSTANCE_ID`

Example:

```bash
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
HF_TOKEN=your_token_here
ENV_BASE_URL=http://127.0.0.1:7860
BENCHMARK=paper_review_env_v1
TASK_IDS=task1,task2,task3,task4
INSTANCE_ID=instance_001
```

## Run locally

### 1. Sync dependencies

```bash
uv sync
```

### 2. Start the server

```bash
uv run uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### 3. Run the baseline

```bash
uv run python inference.py
```

## Docker

Build the image:

```bash
docker build -t paperbench .
```

Run it:

```bash
docker run -p 7860:7860 paperbench
```

Then test:

```bash
curl http://localhost:7860/health
```

## Hugging Face Spaces

This repository is designed to run as a Docker Space.

Use:

* **SDK**: Docker
* **Port**: `7860`
* **License**: Apache 2.0

The Space should respond to `/health`, `/reset`, and `/step`.

## Validation

Before submission, run:

```bash
openenv validate
```

And also check:

```bash
python scripts/validate_fixtures.py
pytest
docker build -t paperbench .
```

## Notes

* The root path `/` is not meant to be a webpage. A `404` at `/` is normal.
* The environment is API-only.
* The graders are deterministic.
* The fixture set is synthetic but designed to model realistic paper screening behavior.
* The baseline is intentionally simple. The environment is the real submission.

## License

Apache 2.0


