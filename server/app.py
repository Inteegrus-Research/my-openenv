# server/app.py
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
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

@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PriorArtBench – Patent Prior‑Art Reasoning</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                max-width: 900px;
                margin: 40px auto;
                padding: 0 20px;
                line-height: 1.6;
                color: #1e1e1e;
                background: #fafafa;
            }
            h1 { color: #0a1172; border-bottom: 2px solid #0a1172; padding-bottom: 10px; }
            h2 { color: #1e3a8a; margin-top: 30px; }
            .badge {
                display: inline-block;
                background: #0a1172;
                color: white;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 14px;
                font-weight: 500;
                margin-right: 8px;
            }
            .endpoint {
                background: white;
                border-radius: 8px;
                padding: 15px 20px;
                margin: 15px 0;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            code {
                background: #e9ecef;
                padding: 2px 6px;
                border-radius: 4px;
                font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
            }
            a {
                color: #0a1172;
                text-decoration: none;
                font-weight: 500;
            }
            a:hover { text-decoration: underline; }
            .footer {
                margin-top: 40px;
                font-size: 14px;
                color: #6b7280;
                border-top: 1px solid #d1d5db;
                padding-top: 20px;
            }
        </style>
    </head>
    <body>
        <h1>🔎 PriorArtBench</h1>
        <p><strong>Patent prior‑art reasoning environment for OpenEnv</strong></p>
        <p>Agents act as patent examiners screening candidate patents under a fixed review budget. 
           Four tasks of increasing difficulty evaluate an agent's ability to triage, rank, and justify prior‑art decisions.</p>

        <h2>📋 Tasks</h2>
        <ul>
            <li><span class="badge">Medium</span> <strong>task1</strong> – Binary relevance screening (20 steps)</li>
            <li><span class="badge">Hard</span> <strong>task2</strong> – Relevance + quality score (24 steps)</li>
            <li><span class="badge">Hard</span> <strong>task3</strong> – Triaging with efficiency bonus (28 steps)</li>
            <li><span class="badge">Very Hard</span> <strong>task4</strong> – Top‑5 ranking with justifications (32 steps)</li>
        </ul>

        <h2>🔌 API Endpoints</h2>
        <div class="endpoint">
            <code>GET /health</code> – Health check<br>
            <code>GET /tasks</code> – List available tasks<br>
            <code>POST /reset</code> – Start a new episode<br>
            <code>POST /step</code> – Submit an action and receive observation + reward
        </div>
        <p>📖 Interactive API docs: <a href="/docs">/docs</a> | <a href="/redoc">/redoc</a></p>

        <h2>📦 Deployment</h2>
        <p>This environment is built for the OpenEnv competition. The Docker image runs on Hugging Face Spaces.</p>
        <p>GitHub: <a href="https://github.com/Inteegrus-Research/AI-openenv-priorartbench" target="_blank">Inteegrus-Research/AI-openenv-priorartbench</a></p>

        <div class="footer">
            PriorArtBench v1.0.0 · OpenEnv compliant · Apache 2.0 License
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


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