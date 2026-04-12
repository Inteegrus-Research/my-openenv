from __future__ import annotations

from typing import Optional, Set

from tasks.task_base import BaseTask

_VALID_LABELS = {"RELEVANT", "NOT_RELEVANT"}


class Task2(BaseTask):
    task_id = "task2"
    budget = 24
    paper_count = 80

    def validate_action(self, action, known_paper_ids: Set[str]) -> Optional[str]:
        err = self._check_paper_id(action, known_paper_ids)
        if err:
            return err
        if getattr(action, "action_type", None) != "review":
            return None
        if getattr(action, "label", None) is None:
            return "label is required for task2"
        return self._check_label(action, _VALID_LABELS)