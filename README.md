---
title: OpenEnv PriorArtBench
emoji: 🔎
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: apache-2.0
app_port: 7860
---

# PriorArtBench

A patent prior‑art reasoning environment built for the OpenEnv Hackathon.  
Agents act as patent examiners screening candidate patents under a fixed review budget.

## What this environment simulates

Patent examiners search for prior art-existing patents or publications that might affect whether a new patent application is granted. This is a real task done daily in patent offices worldwide. The environment captures the core challenge: given a query patent and a large pool of candidates, you must decide which ones are truly relevant before you run out of time (budget).

Each episode presents:
- One **query patent** (the application being examined)
- 20 **candidate patents** (possible prior art)
- A **step budget** (20–32 reviews allowed)

The agent must review candidates one by one and eventually submit a final set of decisions. Performance is graded against ground‑truth relevance scores.

## Observation space

After each step, the environment returns an observation with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | string | Which task is being run (`task1`–`task4`) |
| `task_description` | string | Human‑readable instructions for the task |
| `step` | integer | Number of actions taken so far |
| `budget_remaining` | integer | How many reviews are left before forced submission |
| `query_patent` | object | The patent being examined (id, title, abstract, description, cpc) |
| `candidate_patents` | array | List of all 20 candidate patents (each has id, title, abstract, description, cpc) |
| `decisions_so_far` | object | Map of patent ids to the decisions already made |
| `episode_complete` | boolean | Whether the episode has ended |
| `final_score` | float \| null | Final score (0.0–1.0), only set when `episode_complete` is true |
| `error` | string \| null | Error message if the last action was invalid |

## Action space

Actions are JSON objects. Two types exist:

- **Review a patent** – required fields depend on the task.
- **Submit final decisions** – ends the episode.

### Review action (common fields)

| Field | Type | Description |
|-------|------|-------------|
| `action_type` | string | Must be `"review"` |
| `paper_id` | string | ID of the candidate patent being reviewed |

### Task‑specific review fields

| Task | Additional fields |
|------|-------------------|
| `task1` | `label`: `"RELEVANT"` or `"NOT_RELEVANT"` |
| `task2` | `label`: `"RELEVANT"` or `"NOT_RELEVANT"`<br>`quality_score`: integer 1–4 |
| `task3` | `label`: `"INCLUDE"`, `"EXCLUDE"`, or `"DEFER"` |
| `task4` | `label`: `"INCLUDE"` or `"EXCLUDE"`<br>If `INCLUDE`: `rank` (1–5) and `justification` (string, max 200 chars) |

### Submit action

```json
{"action_type": "submit"}
```

## Tasks

Four tasks of increasing difficulty are provided. Each uses the same patent data but requires different kinds of decisions.

| Task | Difficulty | Budget | Description |
|------|------------|--------|-------------|
| **task1** – Relevance screening | Medium | 20 | Binary choice: relevant or not relevant. Graded by F1 score. |
| **task2** – Novelty‑risk ranking | Hard | 24 | Same binary choice plus a quality score (1–4). Graded by weighted F1 and quality accuracy. |
| **task3** – Claim‑to‑evidence mapping | Hard | 28 | Triaging: INCLUDE, EXCLUDE, or DEFER. Includes an efficiency bonus for using fewer steps. |
| **task4** – Final examiner decision | Very Hard | 32 | Produce a ranked top‑5 shortlist with justifications. Graded by NDCG@5, F1, and justification quality. |

All tasks return a final score between **0.0 and 1.0**. Graders are deterministic and based on the ground‑truth relevance scores in the fixture files.

## Reward function

The environment provides **partial rewards after every review** to help agents learn step‑by‑step:

- Correct binary label: **+0.10** (scaled by relevance strength)
- Incorrect binary label: **–0.05** (scaled)
- Invalid action: **–0.05**
- Task4 includes an extra **+0.05** bonus for justifications that contain relevant technical terms.

The final episode score is still the primary metric, but intermediate rewards make the learning signal much denser.

## Setup and usage

### 1. Install dependencies

```bash
pip install -r requirements.runtime.txt
```

### 2. Generate fixtures (or use the provided ones)

```bash
pip install -r requirements.txt
python scripts/generate_fixtures.py
```

### 3. Start the environment server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### 4. Run the baseline inference script

Create a `.env` file with your credentials (optional; the script works without LLM):

```
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
HF_TOKEN=hf_xxxxxxxxxxxxx
ENV_BASE_URL=http://127.0.0.1:7860
```

Then execute:

```bash
python inference.py
```

The script will output logs in the required `[START]` / `[STEP]` / `[END]` format and print the final scores to stderr.

### 5. Docker (for deployment)

```bash
docker build -t priorart .
docker run -p 7860:7860 priorart
```

The container listens on port 7860 and serves the same FastAPI endpoints.

## Baseline scores

The table below shows reproducible scores obtained with the **heuristic fallback** agent (no LLM required). These scores are deterministic and serve as the official baseline.

| Task | Heuristic score |
|------|-----------------|
| task1 | 0.83 |
| task2 | 0.82 |
| task3 | 0.68 |
| task4 | 0.84 |

*Scores are from `instance_001` of each task. The inference script runs all four tasks in under 2 minutes on a standard machine.*

**Note on LLM scores:** The inference script can optionally use an LLM via the OpenAI client if `HF_TOKEN` is provided. Because LLM performance varies by model and prompt, those scores are not part of the required baseline. The heuristic agent ensures a reproducible lower bound that any LLM‑based agent should exceed.

## File structure

```
.
├── env/                 # Environment core (models, reward, utils)
├── tasks/               # Task definitions and validation
├── graders/             # Episode scoring logic
├── server/              # FastAPI server and session management
├── fixtures/            # JSON files with query patents and ground truth
├── scripts/             # Fixture generation and validation
├── inference.py         # Baseline agent script
├── openenv.yaml         # OpenEnv metadata
├── Dockerfile
└── README.md
```
## License

Apache 2.0 - a permissive, business-friendly open-source license allowing for free use, modification, and distribution of software, including commercial, proprietary products.
