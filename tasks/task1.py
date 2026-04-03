"""
tasks/task1.py
Task 1 — Binary Relevance Classification.

Budget : 12 steps
Papers : ~10 per instance
Labels : RELEVANT | NOT_RELEVANT
Scoring: F1 (binary, RELEVANT as positive class)
"""
from typing import Optional, Set

from env.models import PaperAction
from tasks.task_base import BaseTask

_VALID_LABELS = frozenset({"RELEVANT", "NOT_RELEVANT"})


class Task1(BaseTask):
    task_id = "task1"
    budget = 12
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
                "label is required for task1. "
                "Must be RELEVANT or NOT_RELEVANT."
            )

        return self._check_label(action, _VALID_LABELS)
