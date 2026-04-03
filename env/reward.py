"""
env/reward.py
Episode grading dispatcher.

Calls per-task grader functions from graders/graders.py.
Clamps the returned value to [0.0, 1.0].
Returns 0.0 on any internal error so the environment never crashes here.

Phase 2: grader functions are stubs that return 0.0.
Phase 3: graders/graders.py is replaced with full deterministic implementations.
"""
from typing import Any, Dict


def grade_episode(
    task_id: str,
    completed_decisions: Dict[str, Any],
    fixture: dict,
) -> float:
    """
    Dispatch grading for a completed episode.

    Args:
        task_id:             one of task1 / task2 / task3 / task4
        completed_decisions: paper_id -> action dict, ALL papers present
                             (defaults already applied by the environment)
        fixture:             loaded fixture dict including ground_truth

    Returns:
        float in [0.0, 1.0]
    """
    from graders.graders import grade_task1, grade_task2, grade_task3, grade_task4

    _dispatch = {
        "task1": grade_task1,
        "task2": grade_task2,
        "task3": grade_task3,
        "task4": grade_task4,
    }

    grader_fn = _dispatch.get(task_id)
    if grader_fn is None:
        return 0.0

    try:
        score = grader_fn(completed_decisions, fixture)
        return float(max(0.0, min(1.0, score)))
    except Exception:
        return 0.0
