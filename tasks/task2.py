"""
tasks/task2.py
Task 2 — Relevance + Quality Scoring.

Budget       : 14 steps
Papers       : ~10 per instance
Labels       : RELEVANT | NOT_RELEVANT
quality_score: 1–4 (optional in action; missing ⟹ default 1 at grading time)
Scoring      : 0.6 × relevance_F1 + 0.4 × quality_accuracy
"""
from typing import Optional, Set

from env.models import PaperAction
from tasks.task_base import BaseTask

_VALID_LABELS = frozenset({"RELEVANT", "NOT_RELEVANT"})


class Task2(BaseTask):
    task_id = "task2"
    budget = 14
    paper_count = 10

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
                "label is required for task2. "
                "Must be RELEVANT or NOT_RELEVANT."
            )

        err = self._check_label(action, _VALID_LABELS)
        if err:
            return err

        # quality_score range (1–4) is already enforced by the Pydantic model.
        # No additional check needed here.
        return None
