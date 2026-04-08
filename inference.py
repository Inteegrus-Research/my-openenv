#!/usr/bin/env python3
"""
OpenEnv-PaperBench baseline inference script.

Stdout contract:
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

This script:
- Uses the OpenAI client when HF_TOKEN is available and the provider responds.
- Falls back to a deterministic heuristic policy when the LLM call fails.
- Never repeats the same paper_id within an episode.
- Keeps output format stable and parseable.
- Works locally against the OpenEnv HTTP server.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Optional

import requests
from openai import OpenAI

try:
    from env.models import PaperAction
except Exception:
    PaperAction = None  # type: ignore


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860").rstrip("/")
BENCHMARK = os.getenv("BENCHMARK", "paper_review_env_v1")
TASK_IDS = [t.strip() for t in os.getenv("TASK_IDS", "task1,task2,task3,task4").split(",") if t.strip()]
INSTANCE_ID = os.getenv("INSTANCE_ID", "instance_001")

TEMPERATURE = 0.0
TOP_P = 1.0
MAX_TOKENS = 512
SUCCESS_THRESHOLD = 0.10

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "for", "from", "has",
    "have", "had", "he", "her", "hers", "him", "his", "i", "if", "in", "into", "is", "it",
    "its", "may", "more", "most", "must", "not", "of", "on", "or", "our", "ours", "out",
    "over", "per", "she", "should", "so", "such", "than", "that", "the", "their", "them",
    "themselves", "there", "these", "they", "this", "those", "to", "too", "under", "until",
    "up", "was", "we", "were", "what", "when", "where", "which", "who", "will", "with",
    "without", "you", "your", "yours", "study", "studies", "paper", "papers", "method",
    "methods", "approach", "approaches", "model", "models", "result", "results", "using",
    "use", "used", "based", "through", "new", "proposed", "shown", "show", "showing",
}

POSITIVE_HINTS = {
    "medical", "clinical", "radiology", "imaging", "image", "images", "xray", "x-ray", "ct",
    "mri", "ultrasound", "fundus", "retinal", "pathology", "scan", "segmentation",
    "classification", "diagnosis", "lesion", "tumor", "tumour", "screening", "biomedical",
    "health", "ehr", "sepsis", "oncology", "cancer", "diabetic", "retinopathy", "arrhythmia",
    "cardiac", "cardiology", "brain", "echo", "ecg", "eeg", "dermatology", "histopathology",
    "transformer", "cnn", "unet", "u-net", "vit", "gnn", "gan", "efficientnet", "xgboost",
    "rnn", "lstm", "attention",
}

NEGATIVE_HINTS = {
    "legal", "contract", "trading", "finance", "autonomous", "driving", "lidar", "diffusion",
    "text-to-image", "image-generation", "nas", "edge", "cortex-m", "reinforcement", "policy",
    "order-book", "equity", "nasdaq", "prompt", "llm", "language-model", "summarisation",
    "summarization", "law", "robotics", "vision-language", "vqa", "recommendation",
    "retrieval", "speech", "audio", "navigation", "planning",
}

QUALITY_LOW_HINTS = {
    "editorial", "commentary", "survey", "overview", "opinion", "proposal", "conceptual",
    "preliminary", "pilot", "toy", "proof-of-concept",
}

QUALITY_HIGH_HINTS = {
    "baseline", "baselines", "ablation", "cross-validation", "cross validation",
    "validation", "held-out", "external validation", "prospective", "retrospective",
    "multi-site", "multisite", "code release", "open-source", "open source", "reproducible",
    "reproducibility", "metrics", "auc", "dice", "kappa", "f1", "accuracy", "sensitivity",
    "specificity", "precision", "recall", "mae", "rmse", "auroc", "auprc", "comparison",
    "outperforms", "outperformed", "state-of-the-art", "sota",
}


def _eprint(*args: object) -> None:
    print(*args, file=sys.stderr, flush=True)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def _health_check() -> None:
    resp = requests.get(f"{ENV_BASE_URL}/health", timeout=15)
    resp.raise_for_status()


def _env_reset(task_id: str, instance_id: str) -> Dict[str, Any]:
    resp = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_id": task_id, "instance_id": instance_id},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _env_step(session_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"session_id": session_id, "action": action},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _extract_observation(payload: Dict[str, Any]) -> Dict[str, Any]:
    obs = payload.get("observation")
    if isinstance(obs, dict):
        return obs
    return payload


def _extract_reward(payload: Dict[str, Any]) -> float:
    if isinstance(payload.get("reward"), (int, float)):
        return float(payload["reward"])
    obs = _extract_observation(payload)
    if isinstance(obs.get("reward"), (int, float)):
        return float(obs["reward"])
    return 0.0


def _extract_done(payload: Dict[str, Any]) -> bool:
    if isinstance(payload.get("done"), bool):
        return payload["done"]
    obs = _extract_observation(payload)
    if isinstance(obs.get("episode_complete"), bool):
        return obs["episode_complete"]
    return False


def _extract_error(payload: Dict[str, Any]) -> Optional[str]:
    if isinstance(payload.get("error"), str):
        return payload["error"]
    obs = _extract_observation(payload)
    if isinstance(obs.get("error"), str):
        return obs["error"]
    return None


def _extract_score(payload: Dict[str, Any]) -> float:
    if isinstance(payload.get("score"), (int, float)):
        return float(payload["score"])
    obs = _extract_observation(payload)
    if isinstance(obs.get("final_score"), (int, float)):
        return float(obs["final_score"])
    return 0.0


def _compact_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_json_object(text: str) -> Dict[str, Any]:
    raw = _strip_fences(text)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return {}


def _paper_id(paper: Dict[str, Any]) -> str:
    return str(paper.get("id", ""))


def _paper_text(paper: Dict[str, Any]) -> str:
    parts = [
        paper.get("title", ""),
        paper.get("abstract", ""),
        paper.get("topic_hint", ""),
        paper.get("methodology_hint", ""),
        paper.get("claimed_contribution", ""),
    ]
    return " ".join(str(x) for x in parts if x)


def _tokens(text: str) -> List[str]:
    text = text.lower().replace("-", " ")
    return re.findall(r"[a-z0-9]+", text)


def _content_terms(text: str) -> List[str]:
    return [t for t in _tokens(text) if t not in STOPWORDS and len(t) > 1]


def _has_any(text: str, needles: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(n in lowered for n in needles)


def _relevance_score(task_desc: str, paper: Dict[str, Any]) -> float:
    text = f"{task_desc}\n{_paper_text(paper)}"
    terms = set(_content_terms(text))
    desc_terms = set(_content_terms(task_desc))

    score = 0.0
    score += float(len(terms & desc_terms))

    if _has_any(text, POSITIVE_HINTS):
        score += 2.0
    if _has_any(text, NEGATIVE_HINTS):
        score -= 2.0

    if _has_any(text, {"ct", "mri", "ultrasound", "fundus", "x ray", "xray", "radiology", "pathology"}):
        score += 1.5

    if _has_any(text, {"legal", "trading", "finance", "autonomous driving", "diffusion", "nas"}):
        score -= 1.5

    return score


def _quality_score(paper: Dict[str, Any]) -> int:
    text = _paper_text(paper).lower()
    score = 1

    if _has_any(text, QUALITY_LOW_HINTS):
        score = 1

    if _has_any(text, {"method", "model", "network", "transformer", "cnn", "u-net", "unet", "vit", "gnn", "gan"}):
        score = max(score, 2)

    if _has_any(text, {"metric", "metrics", "auc", "auroc", "auprc", "dice", "f1", "accuracy", "kappa", "sensitivity", "specificity", "precision", "recall", "mae", "rmse"}):
        score = max(score, 3)

    if _has_any(text, QUALITY_HIGH_HINTS):
        score = 4

    return max(1, min(4, score))


def _heuristic_label_task12(task_desc: str, paper: Dict[str, Any]) -> str:
    return "RELEVANT" if _relevance_score(task_desc, paper) >= 2.0 else "NOT_RELEVANT"


def _heuristic_label_task3(task_desc: str, paper: Dict[str, Any]) -> str:
    return "INCLUDE" if _relevance_score(task_desc, paper) >= 2.5 else "EXCLUDE"


def _salient_phrase(paper: Dict[str, Any]) -> str:
    text = _paper_text(paper).lower()
    common_method_words = [
        "transformer", "cnn", "unet", "u-net", "vit", "gnn", "gan", "efficientnet",
        "xgboost", "rnn", "lstm", "attention", "multi-scale", "contrastive",
    ]
    common_metric_words = [
        "auc", "auroc", "auprc", "dice", "f1", "kappa", "accuracy", "sensitivity",
        "specificity", "precision", "recall", "mae", "rmse",
    ]
    method = next((w for w in common_method_words if w in text), None)
    metric = next((w for w in common_metric_words if w in text), None)
    if method and metric:
        return f"{method} with {metric}"
    if method:
        return method
    toks = _content_terms(text)
    if len(toks) >= 2:
        return f"{toks[0]} {toks[1]}"
    if toks:
        return toks[0]
    return "clinical relevance"


def _task4_ranking_score(task_desc: str, paper: Dict[str, Any]) -> float:
    rel = _relevance_score(task_desc, paper)
    qual = _quality_score(paper)
    return rel + 0.6 * qual


def _build_task1_plan(obs: Dict[str, Any]) -> List[Dict[str, Any]]:
    task_desc = obs.get("task_description", "")
    papers = [p for p in obs.get("papers", []) if isinstance(p, dict)]
    plan = [
        {
            "action_type": "review",
            "paper_id": _paper_id(p),
            "label": _heuristic_label_task12(task_desc, p),
        }
        for p in papers
    ]
    plan.append({"action_type": "submit"})
    return plan


def _build_task2_plan(obs: Dict[str, Any]) -> List[Dict[str, Any]]:
    task_desc = obs.get("task_description", "")
    papers = [p for p in obs.get("papers", []) if isinstance(p, dict)]
    plan = []
    for p in papers:
        plan.append(
            {
                "action_type": "review",
                "paper_id": _paper_id(p),
                "label": _heuristic_label_task12(task_desc, p),
                "quality_score": _quality_score(p),
            }
        )
    plan.append({"action_type": "submit"})
    return plan


def _build_task3_plan(obs: Dict[str, Any]) -> List[Dict[str, Any]]:
    task_desc = obs.get("task_description", "")
    papers = [p for p in obs.get("papers", []) if isinstance(p, dict)]
    return [
        {
            "action_type": "review",
            "paper_id": _paper_id(p),
            "label": _heuristic_label_task3(task_desc, p),
        }
        for p in papers
    ] + [{"action_type": "submit"}]


def _build_task4_plan(obs: Dict[str, Any]) -> List[Dict[str, Any]]:
    task_desc = obs.get("task_description", "")
    papers = [p for p in obs.get("papers", []) if isinstance(p, dict)]
    ranked = sorted(
        papers,
        key=lambda p: (-_task4_ranking_score(task_desc, p), _paper_id(p)),
    )[:5]

    plan: List[Dict[str, Any]] = []
    for rank, p in enumerate(ranked, start=1):
        plan.append(
            {
                "action_type": "review",
                "paper_id": _paper_id(p),
                "label": "INCLUDE",
                "rank": rank,
                "justification": (
                    f"{_salient_phrase(p)}; strong validation and clear clinical fit."
                )[:200],
            }
        )
    plan.append({"action_type": "submit"})
    return plan


def _validate_action(action: Dict[str, Any]) -> Dict[str, Any]:
    if PaperAction is None:
        return action
    try:
        return PaperAction.model_validate(action).model_dump(exclude_none=True)
    except Exception:
        return {}


def _maybe_llm_action(client: OpenAI, prompt: str) -> Dict[str, Any]:
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an agent for a paper screening environment. "
                    "Return only a valid JSON object for the next action."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
    )
    content = (completion.choices[0].message.content or "").strip()
    if not content:
        return {}
    return _parse_json_object(content)


def _build_prompt(task_id: str, obs: Dict[str, Any], target_paper: Dict[str, Any]) -> str:
    task_desc = obs.get("task_description", "")
    step = obs.get("step", 0)
    budget_remaining = obs.get("budget_remaining", 0)
    error = obs.get("error")
    decisions = obs.get("decisions_so_far", {})
    target_pid = _paper_id(target_paper)

    lines: List[str] = [
        f"Task: {task_id}",
        f"Step: {step}",
        f"Budget remaining: {budget_remaining}",
        "",
        task_desc,
        "",
        f"Target paper id: {target_pid}",
        "Target paper content:",
        _paper_text(target_paper),
    ]

    if decisions:
        lines.append("")
        lines.append("Decisions so far:")
        for pid in sorted(decisions.keys()):
            lines.append(f"- {pid}: {decisions[pid]}")

    if error:
        lines.append("")
        lines.append(f"Last error: {error}")

    lines += [
        "",
        "Return exactly one JSON object for the target paper only.",
        "Do not include markdown fences or explanation.",
        "Do not invent a different paper_id.",
        "If the task is finished, return {\"action_type\":\"submit\"}.",
    ]
    return "\n".join(lines)


def _run_task(task_id: str, obs: Dict[str, Any], client: Optional[OpenAI], session_id: str) -> float:
    if task_id == "task1":
        plan = _build_task1_plan(obs)
    elif task_id == "task2":
        plan = _build_task2_plan(obs)
    elif task_id == "task3":
        plan = _build_task3_plan(obs)
    else:
        plan = _build_task4_plan(obs)

    rewards: List[float] = []
    steps = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        current_obs = obs

        for planned_action in plan:
            if current_obs.get("episode_complete", False):
                break

            action = dict(planned_action)

            if client is not None and action.get("action_type") == "review":
                try:
                    target_paper = next(
                        (
                            p
                            for p in current_obs.get("papers", [])
                            if isinstance(p, dict) and _paper_id(p) == action.get("paper_id")
                        ),
                        None,
                    )
                    if target_paper is not None:
                        prompt = _build_prompt(task_id, current_obs, target_paper)
                        raw = _maybe_llm_action(client, prompt)
                        if raw:
                            raw["paper_id"] = action["paper_id"]
                            if task_id in {"task1", "task2", "task3"} and "label" in raw:
                                action["label"] = str(raw["label"]).upper()
                            if task_id == "task2" and "quality_score" in raw:
                                try:
                                    action["quality_score"] = max(1, min(4, int(raw["quality_score"])))
                                except Exception:
                                    pass
                            if task_id == "task4" and str(raw.get("label", "EXCLUDE")).upper() == "INCLUDE":
                                action["label"] = "INCLUDE"
                                try:
                                    action["rank"] = max(1, min(5, int(raw.get("rank", action.get("rank", 1)))))
                                except Exception:
                                    pass
                                just = str(raw.get("justification", action.get("justification", ""))).strip()
                                if just:
                                    action["justification"] = just[:200]
                except Exception as exc:
                    _eprint(f"[WARN] task={task_id} optional LLM enhancement skipped: {exc}")

            action = _validate_action(action)
            if not action:
                action = _validate_action(planned_action)
            if not action:
                action = {"action_type": "submit"}

            action_str = _compact_json(action)

            step_payload = _env_step(session_id, action)
            reward = _extract_reward(step_payload)
            done = _extract_done(step_payload)
            error = _extract_error(step_payload)
            current_obs = _extract_observation(step_payload)

            steps += 1
            rewards.append(reward)
            log_step(step=steps, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        score = _extract_score(current_obs)
        if score == 0.0 and rewards:
            score = sum(rewards) / max(len(rewards), 1)
        score = max(0.0, min(1.0, score))
        success = score >= SUCCESS_THRESHOLD

    finally:
        log_end(success=success, steps=steps, score=score, rewards=rewards)

    return score


def main() -> None:
    try:
        _health_check()
    except Exception as exc:
        _eprint(f"[ERROR] environment health check failed: {exc}")
        raise SystemExit(1)

    client: Optional[OpenAI] = None
    if HF_TOKEN:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    scores: List[float] = []
    for task_id in TASK_IDS:
        try:
            reset_payload = _env_reset(task_id, INSTANCE_ID)
            session_id = reset_payload["session_id"]
            obs = _extract_observation(reset_payload)
            scores.append(_run_task(task_id, obs, client, session_id))
        except Exception as exc:
            _eprint(f"[ERROR] task {task_id} failed: {exc}")
            print(f"[END] success=false steps=0 score=0.00 rewards=", flush=True)
            scores.append(0.0)

    if scores:
        _eprint(f"[SUMMARY] mean_score={sum(scores) / len(scores):.4f}")


if __name__ == "__main__":
    main()