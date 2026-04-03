"""
tasks/task_base.py
Abstract base class for all four task configurations.

Responsibilities:
  - load_fixture(): load and minimally validate a JSON fixture from disk
  - validate_action(): task-specific action validation (returns error string or None)
  - shared helpers used by concrete subclasses

Grading logic does NOT belong here.
"""
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Set

from env.models import PaperAction

# Fixtures live at <project_root>/fixtures/
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

# Keys every fixture must contain
_REQUIRED_KEYS = frozenset({"task_description", "papers", "ground_truth"})


class BaseTask(ABC):
    """
    Concrete subclasses set these class attributes:
        task_id    — matches the fixtures/ subdirectory name
        budget     — maximum steps per episode
        paper_count — expected number of papers (reference; not enforced)
    """

    task_id: str = ""
    budget: int = 0
    paper_count: int = 0  # reference only

    # ── Fixture loading ───────────────────────────────────────────────────────

    def load_fixture(self, instance_id: str = "instance_001") -> dict:
        """
        Load a fixture JSON file and perform minimal structural validation.

        Raises:
            FileNotFoundError: if the fixture file does not exist.
            ValueError:        if the file is empty or missing required keys.
        """
        path = FIXTURES_DIR / self.task_id / f"{instance_id}.json"

        if not path.exists():
            raise FileNotFoundError(
                f"Fixture not found: {path}\n"
                f"  task_id={self.task_id!r}, instance_id={instance_id!r}"
            )

        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        if not data:
            raise ValueError(f"Fixture is empty: {path}")

        missing = _REQUIRED_KEYS - set(data.keys())
        if missing:
            raise ValueError(
                f"Fixture {path} is missing required keys: {sorted(missing)}"
            )

        if not isinstance(data["papers"], list) or len(data["papers"]) == 0:
            raise ValueError(
                f"Fixture {path}: 'papers' must be a non-empty list."
            )

        return data

    # ── Action validation contract ────────────────────────────────────────────

    @abstractmethod
    def validate_action(
        self,
        action: PaperAction,
        known_paper_ids: Set[str],
    ) -> Optional[str]:
        """
        Check whether action is valid for this task.

        Returns:
            None            — action is valid; proceed normally.
            str (non-empty) — human-readable error; action must be rejected.

        Must never raise an exception.
        """
        ...

    # ── Shared validation helpers ─────────────────────────────────────────────

    def _check_paper_id(
        self,
        action: PaperAction,
        known_paper_ids: Set[str],
    ) -> Optional[str]:
        """Return an error string if paper_id is not in known_paper_ids."""
        if action.paper_id not in known_paper_ids:
            sample = sorted(known_paper_ids)[:6]
            return (
                f"Unknown paper_id '{action.paper_id}'. "
                f"Known IDs (up to 6): {sample}"
            )
        return None

    def _check_label(
        self,
        action: PaperAction,
        valid_labels: frozenset,
    ) -> Optional[str]:
        """Return an error string if the label is not in valid_labels."""
        if action.label not in valid_labels:
            return (
                f"Invalid label '{action.label}' for {self.task_id}. "
                f"Allowed: {sorted(valid_labels)}"
            )
        return None
