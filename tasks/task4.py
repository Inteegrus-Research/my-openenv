"""
tasks/task4.py
Task 4 — Ranking + Justification Under Budget Pressure.

Budget  : 18 steps  (2 fewer than paper count — structural deficit)
Papers  : ~20 per instance
Labels  : INCLUDE | EXCLUDE

INCLUDE rules (all three required):
  rank          : integer 1–5  (top-5 shortlist position)
  justification : non-empty string, ≤ 200 characters

EXCLUDE rules:
  rank and justification must be absent (or None).

Scoring : 0.50 × nDCG@5  +  0.35 × F1  +  0.15 × justification_validity
"""
from typing import Optional, Set

from env.models import PaperAction
from tasks.task_base import BaseTask

_VALID_LABELS = frozenset({"INCLUDE", "EXCLUDE"})
_MAX_JUSTIFICATION_LEN = 200


class Task4(BaseTask):
    task_id = "task4"
    budget = 18
    paper_count = 20

    def validate_action(
        self,
        action: PaperAction,
        known_paper_ids: Set[str],
    ) -> Optional[str]:
        err = self._check_paper_id(action, known_paper_ids)
        if err:
            return err

        if action.label is None:
            return (
                "label is required for task4. "
                "Must be INCLUDE or EXCLUDE."
            )

        err = self._check_label(action, _VALID_LABELS)
        if err:
            return err

        if action.label == "INCLUDE":
            if action.rank is None:
                return (
                    "rank (1–5) is required when label is INCLUDE in task4."
                )
            # Pydantic already enforces ge=1, le=5, so this is a belt-and-suspenders
            # check for any dict-constructed actions that bypass the model.
            if not (1 <= action.rank <= 5):
                return f"rank must be between 1 and 5, got {action.rank}."

            if not action.justification or action.justification.strip() == "":
                return (
                    "justification is required and must be non-empty "
                    "when label is INCLUDE in task4."
                )
            if len(action.justification) > _MAX_JUSTIFICATION_LEN:
                return (
                    f"justification exceeds {_MAX_JUSTIFICATION_LEN} characters "
                    f"(got {len(action.justification)})."
                )

        return None
