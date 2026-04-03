"""
tasks/task3.py
Task 3 — Adversarial Batch Screening.

Budget  : 15 steps  (= paper count; zero slack for naive strategies)
Papers  : ~15 per instance (includes adversarial red-herring papers)
Labels  : INCLUDE | EXCLUDE | DEFER
          DEFER is treated as EXCLUDE at grading time but can be overwritten
          by a later INCLUDE or EXCLUDE action on the same paper_id.
Scoring : 0.85 × F1  +  0.15 × budget_efficiency
"""
from typing import Optional, Set

from env.models import PaperAction
from tasks.task_base import BaseTask

_VALID_LABELS = frozenset({"INCLUDE", "EXCLUDE", "DEFER"})


class Task3(BaseTask):
    task_id = "task3"
    budget = 15
    paper_count = 15

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
                "label is required for task3. "
                "Must be INCLUDE, EXCLUDE, or DEFER."
            )

        return self._check_label(action, _VALID_LABELS)
