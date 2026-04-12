#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860").rstrip("/")
BENCHMARK = "priorart_bench_v1"
INSTANCE_ID = "instance_001"
TASK_IDS = ["task1", "task2", "task3", "task4"]

MAX_TOKENS = 256
TEMPERATURE = 0.0
SUCCESS_THRESHOLD = 0.10


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = "null" if not error else error.replace("\n", " ").strip()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _health_check() -> None:
    r = requests.get(f"{ENV_BASE_URL}/health", timeout=15)
    r.raise_for_status()


def _reset(task_id: str) -> Dict[str, Any]:
    r = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_id": task_id, "instance_id": INSTANCE_ID},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def _step(session_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"session_id": session_id, "action": action},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def _obs(payload: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(payload.get("observation"), dict):
        return payload["observation"]
    return payload


def _score(payload: Dict[str, Any]) -> float:
    obs = _obs(payload)
    for key in ("final_score", "score"):
        val = obs.get(key, payload.get(key))
        if isinstance(val, (int, float)):
            return float(val)
    return 0.0


def _done(payload: Dict[str, Any]) -> bool:
    if isinstance(payload.get("done"), bool):
        return payload["done"]
    obs = _obs(payload)
    if isinstance(obs.get("episode_complete"), bool):
        return obs["episode_complete"]
    return False


def _error(payload: Dict[str, Any]) -> Optional[str]:
    if isinstance(payload.get("error"), str):
        return payload["error"]
    obs = _obs(payload)
    if isinstance(obs.get("error"), str):
        return obs["error"]
    return None


def _compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(t) >= 4}


def _patent_text(p: Dict[str, Any]) -> str:
    return " ".join(str(p.get(k, "")) for k in ("title", "abstract", "description", "cpc"))


def _similarity(a: str, b: str) -> float:
    ta = _tokens(a)
    tb = _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def _rank_candidates(obs: Dict[str, Any]) -> List[tuple[str, float]]:
    q = _patent_text(obs["query_patent"])
    ranked = []
    for p in obs["candidate_patents"]:
        pid = p["id"]
        txt = _patent_text(p)
        ranked.append((pid, _similarity(q, txt)))
    ranked.sort(key=lambda x: (x[1], x[0]), reverse=True)
    return ranked


def _heuristic_action(task_id: str, obs: Dict[str, Any]) -> Dict[str, Any]:
    ranked = _rank_candidates(obs)
    decisions = obs.get("decisions_so_far", {})
    reviewed = set(decisions.keys())
    budget = int(obs.get("budget_remaining", 0))

    if budget <= 0:
        return {"action_type": "submit"}

    if task_id == "task1":
        pid, score = next(((p, s) for p, s in ranked if p not in reviewed), (ranked[0][0], ranked[0][1]))
        label = "RELEVANT" if score >= 0.08 else "NOT_RELEVANT"
        return {"action_type": "review", "paper_id": pid, "label": label}

    if task_id == "task2":
        pid, score = next(((p, s) for p, s in ranked if p not in reviewed), (ranked[0][0], ranked[0][1]))
        label = "RELEVANT" if score >= 0.08 else "NOT_RELEVANT"
        qs = 4 if score >= 0.20 else 3 if score >= 0.12 else 2 if score >= 0.06 else 1
        return {"action_type": "review", "paper_id": pid, "label": label, "quality_score": qs}

    if task_id == "task3":
        pid, score = next(((p, s) for p, s in ranked if p not in reviewed), (ranked[0][0], ranked[0][1]))
        label = "INCLUDE" if score >= 0.10 else "EXCLUDE"
        return {"action_type": "review", "paper_id": pid, "label": label}

    includes = [pid for pid, _ in ranked[:5] if pid not in reviewed]
    if not includes:
        return {"action_type": "submit"}
    pid = includes[0]
    rank = len([x for x in decisions.values() if isinstance(x, dict) and str(x.get("label", "")).upper() == "INCLUDE"]) + 1
    if rank > 5:
        return {"action_type": "submit"}
    q = _patent_text(obs["query_patent"])
    cand = next(p for p in obs["candidate_patents"] if p["id"] == pid)
    shared = sorted(_tokens(q) & _tokens(_patent_text(cand)))
    just = " ".join(shared[:12]) if shared else "strong prior art overlap"
    return {
        "action_type": "review",
        "paper_id": pid,
        "label": "INCLUDE",
        "rank": rank,
        "justification": just[:200],
    }


def _llm_action(client: OpenAI, task_id: str, obs: Dict[str, Any]) -> Dict[str, Any]:
    candidates = obs.get("candidate_patents", [])
    decisions = obs.get("decisions_so_far", {})
    budget = obs.get("budget_remaining", 0)

    pending = [p for p in candidates if p.get("id") not in decisions]
    if not pending or budget <= 1:
        return {"action_type": "submit"}

    target = pending[0]
    query = obs.get("query_patent", {})

    prompt = f"""
### ROLE: Senior Patent Examiner
### TASK: Determine if the Candidate is Prior Art for the Query.

[QUERY PATENT]
Abstract: {query.get('abstract')}
Field: {query.get('cpc')}

[CANDIDATE PATENT: {target.get('id')}]
Abstract: {target.get('abstract')}

### DECISION CRITERIA:
- RELEVANT/INCLUDE: If technical mechanisms, signal logic, or engineering goals overlap.
- NOT_RELEVANT/EXCLUDE: If the fields of endeavor are entirely unrelated.
- QUALITY SCORE: 1 (Unrelated) to 4 (High Overlap).

### OUTPUT RULES:
- Respond ONLY with a valid JSON object.
- For {task_id}, use the appropriate labels ('RELEVANT' or 'INCLUDE').
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a precise technical auditor. No preamble."},
                {"role": "user", "content": prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = response.choices[0].message.content
        res = json.loads(raw)
        is_include_task = task_id in ("task3", "task4")
        label = str(res.get("label", "")).upper()
        if is_include_task:
            res["label"] = "INCLUDE" if "INCLUDE" in label or "RELEVANT" in label else "EXCLUDE"
        else:
            res["label"] = "RELEVANT" if "INCLUDE" in label or "RELEVANT" in label else "NOT_RELEVANT"
        res["action_type"] = "review"
        res["paper_id"] = target.get("id")
        return res
    except Exception as e:
        _eprint(f"LLM error: {e}")
        return _heuristic_action(task_id, obs)


def _normalize(task_id: str, action: Dict[str, Any], obs: Dict[str, Any]) -> Dict[str, Any]:
    ranked = _rank_candidates(obs)
    reviewed = set(obs.get("decisions_so_far", {}).keys())
    fallback_pid = next((p for p, _ in ranked if p not in reviewed), ranked[0][0])

    if not isinstance(action, dict):
        action = {}
    atype = str(action.get("action_type", "review")).lower()
    if atype == "submit":
        return {"action_type": "submit"}

    pid = str(action.get("paper_id", fallback_pid))
    if pid not in {p["id"] for p in obs["candidate_patents"]}:
        pid = fallback_pid

    if task_id == "task1":
        label = str(action.get("label", "NOT_RELEVANT")).upper()
        if label not in {"RELEVANT", "NOT_RELEVANT"}:
            label = "NOT_RELEVANT"
        return {"action_type": "review", "paper_id": pid, "label": label}

    if task_id == "task2":
        label = str(action.get("label", "NOT_RELEVANT")).upper()
        if label not in {"RELEVANT", "NOT_RELEVANT"}:
            label = "NOT_RELEVANT"
        try:
            q = int(action.get("quality_score", 1))
        except Exception:
            q = 1
        q = max(1, min(4, q))
        return {"action_type": "review", "paper_id": pid, "label": label, "quality_score": q}

    if task_id == "task3":
        label = str(action.get("label", "EXCLUDE")).upper()
        if label not in {"INCLUDE", "EXCLUDE", "DEFER"}:
            label = "EXCLUDE"
        return {"action_type": "review", "paper_id": pid, "label": label}

    label = str(action.get("label", "EXCLUDE")).upper()
    if label not in {"INCLUDE", "EXCLUDE"}:
        label = "EXCLUDE"
    out = {"action_type": "review", "paper_id": pid, "label": label}
    if label == "INCLUDE":
        try:
            rank = int(action.get("rank", 1))
        except Exception:
            rank = 1
        rank = max(1, min(5, rank))
        just = str(action.get("justification", "")).strip()
        if not just:
            just = "strong prior art evidence"
        out["rank"] = rank
        out["justification"] = just[:200]
    return out


def _run_task(task_id: str, client: Optional[OpenAI]) -> float:
    reset_payload = _reset(task_id)
    session_id = reset_payload["session_id"]
    obs = _obs(reset_payload)

    rewards: List[float] = []
    steps = 0
    final_score = 0.0
    success = False

    log_start(task_id, BENCHMARK, MODEL_NAME)

    try:
        while not obs.get("episode_complete", False):
            if client is not None:
                try:
                    raw_action = _llm_action(client, task_id, obs)
                except Exception:
                    raw_action = {}
            else:
                raw_action = {}

            action = _normalize(task_id, raw_action if raw_action else _heuristic_action(task_id, obs), obs)
            action_json = _compact(action)

            step_payload = _step(session_id, action)
            obs = _obs(step_payload)

            reward = float(step_payload.get("reward", 0.0) or 0.0)
            done = bool(step_payload.get("done", obs.get("episode_complete", False)))
            err = _error(step_payload)

            steps += 1
            rewards.append(reward)
            log_step(steps, action_json, reward, done, err)

            if done:
                break

        final_score = max(0.01, min(0.99, _score(obs if isinstance(obs, dict) else step_payload)))
        success = final_score >= SUCCESS_THRESHOLD

    finally:
        log_end(success, steps, final_score, rewards)

    return final_score


def main() -> None:
    _health_check()

    client = None
    if OpenAI is not None and HF_TOKEN:
        try:
            client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        except Exception as exc:
            _eprint(f"OpenAI client disabled: {exc}")

    scores = []
    for task_id in TASK_IDS:
        try:
            score = _run_task(task_id, client)
            scores.append(score)
        except Exception as exc:
            _eprint(f"Task {task_id} failed: {exc}")
            scores.append(0.01)

    if scores:
        _eprint(f"Mean score: {sum(scores)/len(scores):.4f}")


if __name__ == "__main__":
    main()