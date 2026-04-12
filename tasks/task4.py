from __future__ import annotations

from typing import Optional, Set

from tasks.task_base import BaseTask

_VALID_LABELS = {"INCLUDE", "EXCLUDE"}


class Task4(BaseTask):
    task_id = "task4"
    budget = 32
    paper_count = 80

    def validate_action(self, action, known_paper_ids: Set[str]) -> Optional[str]:
        err = self._check_paper_id(action, known_paper_ids)
        if err:
            return err
        if getattr(action, "action_type", None) != "review":
            return None
        if getattr(action, "label", None) is None:
            return "label is required for task4"
        err = self._check_label(action, _VALID_LABELS)
        if err:
            return err
        if getattr(action, "label", None) == "INCLUDE":
            if getattr(action, "rank", None) is None:
                return "rank is required when label is INCLUDE"
            if getattr(action, "justification", None) is None or not str(action.justification).strip():
                return "justification is required when label is INCLUDE"
        return None