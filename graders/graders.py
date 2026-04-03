"""
graders/graders.py
Per-task grader functions.

Each function is a pure function:
    grade_taskN(completed_decisions, fixture) -> float in [0.0, 1.0]

Args:
    completed_decisions : Dict[paper_id -> action_dict]
                          Every paper in the fixture is present
                          (defaults have been applied by env/reward.py).
    fixture             : The loaded fixture dict, including "ground_truth".

Phase 2: All functions are stubs returning 0.0.
Phase 3: Replace stubs with full deterministic implementations:
    grade_task1  — binary F1 (RELEVANT as positive class)
    grade_task2  — 0.6×relevance_F1 + 0.4×quality_accuracy
    grade_task3  — 0.85×F1 + 0.15×budget_efficiency
    grade_task4  — 0.50×nDCG@5 + 0.35×F1 + 0.15×justification_validity
"""
from typing import Any, Dict


def grade_task1(
    completed_decisions: Dict[str, Any],
    fixture: dict,
) -> float:
    """Task 1: Binary relevance F1.  Phase 2 stub."""
    return 0.0


def grade_task2(
    completed_decisions: Dict[str, Any],
    fixture: dict,
) -> float:
    """Task 2: Weighted F1 + quality accuracy.  Phase 2 stub."""
    return 0.0


def grade_task3(
    completed_decisions: Dict[str, Any],
    fixture: dict,
) -> float:
    """Task 3: F1 + budget efficiency.  Phase 2 stub."""
    return 0.0


def grade_task4(
    completed_decisions: Dict[str, Any],
    fixture: dict,
) -> float:
    """Task 4: nDCG@5 + F1 + justification validity.  Phase 2 stub."""
    return 0.0
