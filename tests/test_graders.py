from __future__ import annotations

import json
from pathlib import Path

import pytest

from graders.grader1 import grade_task1
from graders.grader2 import grade_task2
from graders.grader3 import grade_task3
from graders.grader4 import grade_task4

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"


def load_fixture(task_id: str, instance_id: str = "instance_001") -> dict:
    path = FIXTURES / task_id / f"{instance_id}.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def all_negative(task_id: str, fixture: dict) -> dict:
    ids = [p["id"] for p in fixture["candidate_patents"]]
    if task_id == "task1":
        return {pid: {"action_type": "review", "paper_id": pid, "label": "NOT_RELEVANT"} for pid in ids}
    if task_id == "task2":
        return {pid: {"action_type": "review", "paper_id": pid, "label": "NOT_RELEVANT", "quality_score": 1} for pid in ids}
    if task_id == "task3":
        return {pid: {"action_type": "review", "paper_id": pid, "label": "EXCLUDE"} for pid in ids}
    return {pid: {"action_type": "review", "paper_id": pid, "label": "EXCLUDE"} for pid in ids}


def all_positive(task_id: str, fixture: dict) -> dict:
    ids = [p["id"] for p in fixture["candidate_patents"]]
    if task_id == "task1":
        return {pid: {"action_type": "review", "paper_id": pid, "label": "RELEVANT"} for pid in ids}
    if task_id == "task2":
        return {pid: {"action_type": "review", "paper_id": pid, "label": "RELEVANT", "quality_score": 4} for pid in ids}
    if task_id == "task3":
        return {pid: {"action_type": "review", "paper_id": pid, "label": "INCLUDE"} for pid in ids}
    return {
        pid: {
            "action_type": "review",
            "paper_id": pid,
            "label": "INCLUDE",
            "rank": min(i + 1, 5),
            "justification": "strong prior art evidence",
        }
        for i, pid in enumerate(ids)
    }


@pytest.mark.parametrize("task_id", ["task1", "task2", "task3", "task4"])
def test_grader_scores_in_range(task_id: str) -> None:
    fixture = load_fixture(task_id)
    for fn in [grade_task1, grade_task2, grade_task3, grade_task4]:
        if task_id == "task1":
            score = grade_task1(all_negative(task_id, fixture), fixture)
        elif task_id == "task2":
            score = grade_task2(all_negative(task_id, fixture), fixture)
        elif task_id == "task3":
            score = grade_task3(all_negative(task_id, fixture), fixture, steps_used=10)
        else:
            score = grade_task4(all_negative(task_id, fixture), fixture)
        assert 0.0 < score < 1.0


@pytest.mark.parametrize("task_id", ["task1", "task2", "task3", "task4"])
def test_grader_non_constant(task_id: str) -> None:
    fixture = load_fixture(task_id)
    if task_id == "task1":
        s1 = grade_task1(all_negative(task_id, fixture), fixture)
        s2 = grade_task1(all_positive(task_id, fixture), fixture)
    elif task_id == "task2":
        s1 = grade_task2(all_negative(task_id, fixture), fixture)
        s2 = grade_task2(all_positive(task_id, fixture), fixture)
    elif task_id == "task3":
        s1 = grade_task3(all_negative(task_id, fixture), fixture, steps_used=10)
        s2 = grade_task3(all_positive(task_id, fixture), fixture, steps_used=10)
    else:
        s1 = grade_task4(all_negative(task_id, fixture), fixture)
        s2 = grade_task4(all_positive(task_id, fixture), fixture)
    assert abs(s1 - s2) > 0.01


def test_task4_uses_justification() -> None:
    fixture = load_fixture("task4")
    ids = [p["id"] for p in fixture["candidate_patents"]]
    pid = ids[0]
    base = {p["id"]: {"action_type": "review", "paper_id": p["id"], "label": "EXCLUDE"} for p in fixture["candidate_patents"]}
    with_just = dict(base)
    with_just[pid] = {
        "action_type": "review",
        "paper_id": pid,
        "label": "INCLUDE",
        "rank": 1,
        "justification": "strong prior art evidence in patent language",
    }
    without_just = dict(base)
    without_just[pid] = {
        "action_type": "review",
        "paper_id": pid,
        "label": "INCLUDE",
        "rank": 1,
        "justification": "zzz",
    }
    s1 = grade_task4(with_just, fixture)
    s2 = grade_task4(without_just, fixture)
    assert s1 >= s2
