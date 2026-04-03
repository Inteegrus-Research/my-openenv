"""
env/environment.py
PaperReviewEnv — the core OpenEnv-PaperBench environment.

Public interface (OpenEnv-compatible):
    reset(task_id, instance_id="instance_001") -> Observation
    step(action: PaperAction)                  -> Observation
    state()                                    -> Observation

Contract:
    - Deterministic: same inputs always produce the same outputs.
    - Never crashes on malformed action input.
    - Invalid actions cost a step, set obs.error, leave decisions unchanged.
    - Valid DEFER (task3) can be overwritten by a later action on the same paper.
    - Episode closes on submit action or budget exhaustion.
    - Terminal observation has episode_complete=True and final_score populated.
"""
from typing import Any, Dict, Optional

from env.models import Observation, PaperAction, PaperRecord
from env.utils import TASK_IDS, apply_defaults, sorted_decisions
from tasks.task1 import Task1
from tasks.task2 import Task2
from tasks.task3 import Task3
from tasks.task4 import Task4

_TASK_REGISTRY: Dict[str, Any] = {
    "task1": Task1,
    "task2": Task2,
    "task3": Task3,
    "task4": Task4,
}


class PaperReviewEnv:
    """
    Sequential paper screening environment.

    Typical usage:
        env = PaperReviewEnv()
        obs = env.reset("task1")
        while not obs.episode_complete:
            action = agent.decide(obs)
            obs = env.step(action)
        print(obs.final_score)
    """

    def __init__(self) -> None:
        self._task: Optional[Any] = None
        self._task_id: Optional[str] = None
        self._fixture: Optional[dict] = None
        self._step: int = 0
        self._budget_remaining: int = 0
        self._decisions: Dict[str, dict] = {}
        self._episode_complete: bool = False
        self._final_score: Optional[float] = None
        self._last_error: Optional[str] = None

    # ── Public interface ──────────────────────────────────────────────────────

    def reset(
        self,
        task_id: str,
        instance_id: str = "instance_001",
    ) -> Observation:
        """
        Initialise a new episode.

        Args:
            task_id:     one of "task1" / "task2" / "task3" / "task4"
            instance_id: fixture file stem, e.g. "instance_001"

        Returns:
            Initial Observation with step=0, full paper list, empty decisions.

        Raises:
            ValueError:        unknown task_id
            FileNotFoundError: fixture file does not exist
            ValueError:        fixture is structurally invalid
        """
        if task_id not in _TASK_REGISTRY:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Valid options: {sorted(_TASK_REGISTRY)}"
            )

        task_cls = _TASK_REGISTRY[task_id]
        self._task = task_cls()
        self._task_id = task_id
        self._fixture = self._task.load_fixture(instance_id)
        self._step = 0
        self._budget_remaining = self._task.budget
        self._decisions = {}
        self._episode_complete = False
        self._final_score = None
        self._last_error = None

        return self._build_observation()

    def step(self, action: PaperAction) -> Observation:
        """
        Process one action and return the updated observation.

        Rules:
          - Budget is decremented by 1 for every call (valid or invalid).
          - Invalid actions set obs.error and do NOT update decisions_so_far.
          - Valid actions update decisions_so_far (overwrites any prior action
            for the same paper_id, enabling DEFER→INCLUDE/EXCLUDE in task3).
          - A "submit" action closes the episode immediately.
          - Budget reaching 0 closes the episode automatically.
          - Calling step() on a closed episode returns the terminal observation
            unchanged (no budget decrement, no state change).

        Raises:
            RuntimeError: if reset() has not been called.
        """
        if self._task is None:
            raise RuntimeError(
                "Environment is not initialised. Call reset() before step()."
            )

        # Episode already closed — return terminal observation, no state change.
        if self._episode_complete:
            return self._build_observation()

        self._step += 1
        self._budget_remaining -= 1
        self._last_error = None

        try:
            if action.action_type == "submit":
                self._close_episode()
                return self._build_observation()

            # action_type == "review"
            known_ids = {p["id"] for p in self._fixture["papers"]}
            error_msg = self._task.validate_action(action, known_ids)

            if error_msg:
                self._last_error = error_msg
            else:
                # Store as plain dict; exclude None fields for clean serialisation.
                self._decisions[action.paper_id] = action.model_dump(
                    exclude_none=True
                )

        except Exception as exc:  # pragma: no cover
            self._last_error = f"Unexpected error processing action: {exc}"

        # Budget exhausted → close episode
        if self._budget_remaining <= 0 and not self._episode_complete:
            self._close_episode()

        return self._build_observation()

    def state(self) -> Observation:
        """
        Return the current observation without modifying any state.
        Idempotent; safe to call at any point after reset().

        Raises:
            RuntimeError: if reset() has not been called.
        """
        if self._task is None:
            raise RuntimeError(
                "Environment is not initialised. Call reset() before state()."
            )
        return self._build_observation()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _close_episode(self) -> None:
        """Mark episode complete and compute the final score."""
        self._episode_complete = True
        self._final_score = self._compute_score()

    def _compute_score(self) -> float:
        """
        Apply defaults to unreviewed papers then call the grader dispatcher.
        Returns 0.0 on any error — the environment must never crash here.
        """
        from env.reward import grade_episode

        paper_ids = [p["id"] for p in self._fixture["papers"]]
        completed = apply_defaults(paper_ids, self._decisions, self._task_id)
        return grade_episode(self._task_id, completed, self._fixture)

    def _build_observation(self) -> Observation:
        """
        Construct an Observation from the current state.
        Paper order follows the fixture. Decisions are sorted by paper_id.
        """
        papers = [PaperRecord(**p) for p in self._fixture["papers"]]
        return Observation(
            task_id=self._task_id,
            task_description=self._fixture.get("task_description", ""),
            step=self._step,
            budget_remaining=self._budget_remaining,
            papers=papers,
            decisions_so_far=sorted_decisions(self._decisions),
            episode_complete=self._episode_complete,
            final_score=self._final_score,
            error=self._last_error,
        )
