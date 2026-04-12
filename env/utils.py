from __future__ import annotations

from typing import Any, Dict

TASK_BUDGETS: Dict[str, int] = {
    "task1": 20,
    "task2": 24,
    "task3": 28,
    "task4": 32,
}

TASK_PAPER_COUNTS: Dict[str, int] = {
    "task1": 80,
    "task2": 80,
    "task3": 80,
    "task4": 80,
}

VALID_LABELS: Dict[str, frozenset[str]] = {
    "task1": frozenset({"RELEVANT", "NOT_RELEVANT"}),
    "task2": frozenset({"RELEVANT", "NOT_RELEVANT"}),
    "task3": frozenset({"INCLUDE", "EXCLUDE", "DEFER"}),
    "task4": frozenset({"INCLUDE", "EXCLUDE"}),
}

DEFAULT_LABELS: Dict[str, str] = {
    "task1": "NOT_RELEVANT",
    "task2": "NOT_RELEVANT",
    "task3": "EXCLUDE",
    "task4": "EXCLUDE",
}

TASK_IDS = frozenset(TASK_BUDGETS.keys())


def sorted_decisions(decisions: Dict[str, Any]) -> Dict[str, Any]:
    return dict(sorted(decisions.items(), key=lambda x: x[0]))


def get_default_label(task_id: str) -> str:
    return DEFAULT_LABELS.get(task_id, "EXCLUDE")


def apply_defaults(
    patent_ids: list[str],
    decisions: Dict[str, Any],
    task_id: str,
) -> Dict[str, Any]:
    out = dict(decisions)
    default_label = get_default_label(task_id)
    for pid in sorted(patent_ids):
        if pid not in out:
            out[pid] = {
                "action_type": "review",
                "paper_id": pid,
                "label": default_label,
            }
    return out