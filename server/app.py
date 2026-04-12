# server/app.py
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from server.session import session_store

app = FastAPI(
    title="PriorArtBench",
    version="1.0.0",
    description="Patent prior-art reasoning environment",
)


class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    instance_id: Optional[str] = None


class StepRequest(BaseModel):
    session_id: str
    action: Dict[str, Any]


_TASK_DESCRIPTORS = [
    {"task_id": "task1", "name": "Prior-art relevance screening", "budget": 20, "paper_count": 80},
    {"task_id": "task2", "name": "Novelty-risk ranking", "budget": 24, "paper_count": 80},
    {"task_id": "task3", "name": "Claim-to-evidence mapping", "budget": 28, "paper_count": 80},
    {"task_id": "task4", "name": "Final examiner decision", "budget": 32, "paper_count": 80},
]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "env": "priorart_bench_v1"}


@app.get("/tasks")
def tasks() -> list[dict[str, Any]]:
    return _TASK_DESCRIPTORS


@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    from env.environment import PaperReviewEnv
    from env.models import PaperAction, Observation, Reward

    env = PaperReviewEnv()
    task_id = req.task_id if req and req.task_id else "task1"
    instance_id = req.instance_id if req and req.instance_id else "instance_001"
    try:
        obs = env.reset(task_id=task_id, instance_id=instance_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    session_id = session_store.create(env)
    return {"session_id": session_id, "observation": obs.model_dump()}


@app.post("/step")
def step(req: StepRequest):
    from env.models import PaperAction

    env = session_store.get(req.session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="session not found")
    try:
        action = PaperAction.model_validate(req.action)
        obs, reward, done, info = env.step(action)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {
        "observation": obs.model_dump(),
        "reward": reward.value,
        "done": done,
        "error": info.get("error") if done or info.get("error") else None,
    }


def main() -> None:
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()