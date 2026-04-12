from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Set

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


class BaseTask(ABC):
    task_id: str = ""
    budget: int = 0
    paper_count: int = 0

    def load_fixture(self, instance_id: str = "instance_001") -> dict[str, Any]:
        path = FIXTURES_DIR / self.task_id / f"{instance_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Fixture not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Fixture must be a JSON object: {path}")

        required = {"task_description", "query_patent", "candidate_patents", "ground_truth"}
        missing = required - set(data.keys())
        if missing:
            raise ValueError(f"Fixture {path} missing keys: {sorted(missing)}")

        if not isinstance(data["candidate_patents"], list) or not data["candidate_patents"]:
            raise ValueError(f"Fixture {path} has empty candidate_patents")

        return data

    @abstractmethod
    def validate_action(self, action: Any, known_paper_ids: Set[str]) -> Optional[str]:
        raise NotImplementedError

    def _check_paper_id(self, action: Any, known_paper_ids: Set[str]) -> Optional[str]:
        paper_id = getattr(action, "paper_id", None)
        if paper_id not in known_paper_ids:
            return f"Unknown paper_id '{paper_id}'"
        return None

    def _check_label(self, action: Any, allowed: Set[str]) -> Optional[str]:
        label = getattr(action, "label", None)
        if label not in allowed:
            return f"Invalid label '{label}' for {self.task_id}. Allowed: {sorted(allowed)}"
        return None