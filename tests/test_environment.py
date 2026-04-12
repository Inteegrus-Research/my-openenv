from __future__ import annotations

import json
from pathlib import Path

import pytest

from env.environment import PaperReviewEnv
from env.models import PaperAction
from env.utils import TASK_BUDGETS

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"


def load_fixture(task_id: str, instance_id: str = "instance_001") -> dict:
    path = FIXTURES / task_id / f"{instance_id}.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize("task_id", ["task1", "task2", "task3", "task4"])
def test_reset_state_cycle(task_id: str) -> None:
    env = PaperReviewEnv()
    obs = env.reset(task_id=task_id, instance_id="instance_001")

    assert obs.task_id == task_id
    assert obs.step == 0
    assert obs.budget_remaining == TASK_BUDGETS[task_id]
    assert obs.episode_complete is False
    assert obs.final_score is None
    assert obs.error is None
    assert len(obs.candidate_patents) > 0

    state = env.state()
    assert state.model_dump() == obs.model_dump()


@pytest.mark.parametrize("task_id", ["task1", "task2", "task3", "task4"])
def test_submit_ends_episode(task_id: str) -> None:
    env = PaperReviewEnv()
    obs = env.reset(task_id=task_id, instance_id="instance_001")
    out = env.step(PaperAction(action_type="submit"))

    assert out.episode_complete is True
    assert out.final_score is not None
    assert 0.0 < out.final_score < 1.0
    assert out.step == 1
    assert out.budget_remaining == TASK_BUDGETS[task_id] - 1


@pytest.mark.parametrize("task_id,label", [
    ("task1", "RELEVANT"),
    ("task2", "RELEVANT"),
    ("task3", "INCLUDE"),
    ("task4", "INCLUDE"),
])
def test_step_accepts_valid_review(task_id: str, label: str) -> None:
    env = PaperReviewEnv()
    obs = env.reset(task_id=task_id, instance_id="instance_001")
    pid = obs.candidate_patents[0].id

    action = {"action_type": "review", "paper_id": pid, "label": label}
    if task_id == "task2":
        action["quality_score"] = 3
    if task_id == "task4":
        action["rank"] = 1
        action["justification"] = "strong patent prior art evidence"

    out = env.step(action)

    assert out.step == 1
    assert out.error is None or isinstance(out.error, str)
    assert pid in out.decisions_so_far


@pytest.mark.parametrize("task_id", ["task1", "task2", "task3", "task4"])
def test_invalid_action_handled_safely(task_id: str) -> None:
    env = PaperReviewEnv()
    obs = env.reset(task_id=task_id, instance_id="instance_001")

    out = env.step({"action_type": "review", "paper_id": "missing", "label": "RELEVANT"})

    assert out.step == 1
    assert out.error is not None
    assert out.episode_complete is False
    assert out.final_score is None


@pytest.mark.parametrize("task_id", ["task1", "task2", "task3", "task4"])
def test_budget_exhaustion_ends_episode(task_id: str) -> None:
    env = PaperReviewEnv()
    obs = env.reset(task_id=task_id, instance_id="instance_001")
    pid = obs.candidate_patents[0].id

    for _ in range(TASK_BUDGETS[task_id]):
        action = {"action_type": "review", "paper_id": pid, "label": "EXCLUDE"}
        if task_id == "task1":
            action["label"] = "NOT_RELEVANT"
        elif task_id == "task2":
            action["label"] = "NOT_RELEVANT"
            action["quality_score"] = 1
        elif task_id == "task3":
            action["label"] = "EXCLUDE"
        else:
            action["label"] = "EXCLUDE"
        out = env.step(action)

    assert out.episode_complete is True
    assert out.final_score is not None
    assert 0.0 < out.final_score < 1.0


def test_state_does_not_mutate() -> None:
    env = PaperReviewEnv()
    obs = env.reset("task1", "instance_001")
    state1 = env.state()
    state2 = env.state()
    assert state1.model_dump() == state2.model_dump()
    assert obs.model_dump() == state1.model_dump()
