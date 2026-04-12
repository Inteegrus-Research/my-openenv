from __future__ import annotations

from typing import Any, Dict, Optional

from env.utils import TASK_BUDGETS
from graders.grader1 import _clamp_open, _f1, _paper_ids, _top_k_positive_ids


def grade_task3(
    completed_decisions: Dict[str, Any],
    fixture: dict,
    steps_used: Optional[int] = None,
    **kwargs,
) -> float:
    try:
        ids = _paper_ids(fixture)
        positives = _top_k_positive_ids(fixture, k=max(10, len(ids) // 8))
        y_true = [1 if pid in positives else 0 for pid in ids]
        y_pred = []
        for pid in ids:
            label = str(completed_decisions.get(pid, {}).get("label", "EXCLUDE")).upper()
            if label == "INCLUDE":
                y_pred.append(1)
            else:
                y_pred.append(0)

        f1 = _f1(y_true, y_pred)
        budget = TASK_BUDGETS.get("task3", 28)
        if steps_used is None:
            steps_used = min(budget, len([v for v in completed_decisions.values() if isinstance(v, dict)]))
        efficiency = max(0.0, min(1.0, (budget - steps_used) / max(1, budget)))
        return _clamp_open(0.85 * f1 + 0.15 * efficiency)
    except Exception:
        return 0.01