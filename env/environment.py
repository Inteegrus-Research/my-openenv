from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from env.models import Observation, PaperAction, PatentRecord, Reward
from env.reward import grade_episode
from env.utils import TASK_BUDGETS, apply_defaults, sorted_decisions
from tasks.task1 import Task1
from tasks.task2 import Task2
from tasks.task3 import Task3
from tasks.task4 import Task4

_TASK_REGISTRY = {
    "task1": Task1,
    "task2": Task2,
    "task3": Task3,
    "task4": Task4,
}

TASK_DESCRIPTIONS = {
    "task1": "Screen candidate patents for relevance to the query patent. Mark the strongest prior-art candidates as RELEVANT and the rest as NOT_RELEVANT.",
    "task2": "Rank candidate patents by novelty risk and provide a quality score from 1 to 4 for each review decision.",
    "task3": "Decide whether each candidate should be INCLUDED, EXCLUDED, or DEFERRED. DEFER counts as EXCLUDE at grading time.",
    "task4": "Produce a ranked top-5 prior-art shortlist with justification strings for INCLUDE decisions. This is the hardest task.",
}


class PaperReviewEnv:
    def __init__(self) -> None:
        self._task: Optional[Any] = None
        self._task_id: Optional[str] = None
        self._fixture: Optional[dict] = None
        self._step_count = 0
        self._budget_remaining = 0
        self._decisions: Dict[str, Dict[str, Any]] = {}
        self._episode_complete = False
        self._final_score: Optional[float] = None
        self._last_error: Optional[str] = None
        self._last_action_correct: Optional[bool] = None

    def reset(self, *, task_id: str = "task1", instance_id: str = "instance_001") -> Observation:
        if task_id not in _TASK_REGISTRY:
            task_id = "task1"
        task_cls = _TASK_REGISTRY[task_id]
        self._task = task_cls()
        self._task_id = task_id
        self._fixture = self._task.load_fixture(instance_id or "instance_001")
        self._step_count = 0
        self._budget_remaining = TASK_BUDGETS.get(task_id, self._task.budget)
        self._decisions = {}
        self._episode_complete = False
        self._final_score = None
        self._last_error = None
        self._last_action_correct = None
        return self._build_observation()

    def step(self, action: PaperAction) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self._task is None or self._fixture is None or self._task_id is None:
            raise RuntimeError("Call reset() before step().")

        if self._episode_complete:
            obs = self._build_observation()
            return obs, Reward(value=0.0), True, {"error": self._last_error}

        self._step_count += 1
        self._budget_remaining -= 1
        self._last_error = None
        self._last_action_correct = None

        try:
            if action.action_type == "submit":
                self._close_episode()
                reward_val = 0.0
            else:
                known_ids = {p["id"] for p in self._fixture["candidate_patents"]}
                error = self._task.validate_action(action, known_ids)
                if error:
                    self._last_error = error
                    reward_val = -0.05
                else:
                    self._decisions[action.paper_id] = action.model_dump(exclude_none=True)
                    reward_val = self._compute_partial_reward(action)
                    self._last_action_correct = reward_val > 0
        except Exception as exc:
            self._last_error = str(exc)
            reward_val = -0.05

        if self._budget_remaining <= 0:
            self._close_episode()

        obs = self._build_observation()
        info = {"error": self._last_error}
        return obs, Reward(value=reward_val), self._episode_complete, info

    def state(self) -> Observation:
        if self._task is None or self._fixture is None or self._task_id is None:
            raise RuntimeError("Call reset() before state().")
        return self._build_observation()

    def _close_episode(self) -> None:
        self._episode_complete = True
        self._final_score = self._compute_score()

    def _compute_score(self) -> float:
        assert self._fixture is not None
        assert self._task_id is not None
        patent_ids = [p["id"] for p in self._fixture["candidate_patents"]]
        completed = apply_defaults(patent_ids, self._decisions, self._task_id)
        return grade_episode(
            self._task_id,
            completed,
            self._fixture,
            steps_used=self._step_count,
        )

    def _compute_partial_reward(self, action: PaperAction) -> float:
        if action.action_type != "review" or not action.paper_id:
            return 0.0
        pid = action.paper_id
        gt_scores = self._fixture["ground_truth"]["relevance_scores"]
        gt_label = gt_scores.get(pid, 0.0) > 0.5

        if self._task_id in ("task1", "task2"):
            pred_label = action.label == "RELEVANT"
            correct = (gt_label == pred_label)
            base = 0.1 if correct else -0.05
            strength = abs(gt_scores.get(pid, 0.5) - 0.5) * 2
            return base * (0.5 + 0.5 * strength)
        elif self._task_id == "task3":
            pred_include = action.label == "INCLUDE"
            correct = (gt_label == pred_include)
            base = 0.1 if correct else -0.05
            strength = abs(gt_scores.get(pid, 0.5) - 0.5) * 2
            return base * (0.5 + 0.5 * strength)
        elif self._task_id == "task4":
            pred_include = action.label == "INCLUDE"
            correct = (gt_label == pred_include)
            reward = 0.1 if correct else -0.05
            if correct and pred_include:
                from graders.grader1 import _justification_ok
                vocab = set()
                q = self._fixture["query_patent"]
                for field in ("abstract", "description", "cpc"):
                    vocab.update([t for t in __import__("re").findall(r"[a-zA-Z0-9]+", str(q.get(field, "")).lower()) if len(t) >= 5])
                if _justification_ok(action.justification, vocab):
                    reward += 0.05
            return reward
        return 0.0

    def _build_observation(self) -> Observation:
        assert self._fixture is not None
        assert self._task_id is not None

        query = PatentRecord.model_validate(self._fixture["query_patent"])
        candidates = [PatentRecord.model_validate(p) for p in self._fixture["candidate_patents"]]
        task_desc = TASK_DESCRIPTIONS.get(self._task_id, self._fixture.get("task_description", ""))

        return Observation(
            task_id=self._task_id,
            task_description=task_desc,
            step=self._step_count,
            budget_remaining=self._budget_remaining,
            query_patent=query,
            candidate_patents=candidates,
            decisions_so_far=sorted_decisions(self._decisions),
            episode_complete=self._episode_complete,
            final_score=self._final_score,
            error=self._last_error,
        )