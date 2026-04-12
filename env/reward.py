from __future__ import annotations

from typing import Any, Dict

from env.utils import TASK_BUDGETS


def grade_episode(
    task_id: str,
    completed_decisions: Dict[str, Any],
    fixture: dict,
    *,
    steps_used: int | None = None,
) -> float:
    from graders.grader1 import grade_task1
    from graders.grader2 import grade_task2
    from graders.grader3 import grade_task3
    from graders.grader4 import grade_task4

    dispatch = {
        "task1": grade_task1,
        "task2": grade_task2,
        "task3": grade_task3,
        "task4": grade_task4,
    }

    grader = dispatch.get(task_id)
    if grader is None:
        return 0.01

    try:
        if task_id == "task3":
            score = grader(completed_decisions, fixture, steps_used=steps_used)
        else:
            score = grader(completed_decisions, fixture)
        return max(0.01, min(0.99, float(score)))
    except Exception:
        return 0.01