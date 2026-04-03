"""
tests/test_env.py
Phase 2 test suite.

Covers:
  - Model construction and validation (PaperAction, Observation, PaperRecord)
  - reset() returns a valid Observation for all four tasks
  - step() decrements budget correctly
  - step() stores valid decisions in decisions_so_far
  - step() rejects invalid actions (sets error, does NOT update decisions)
  - step() handles submit action — closes episode, populates final_score
  - Budget exhaustion closes the episode automatically
  - state() mirrors current internal state
  - Calling step() on a closed episode is a no-op
  - apply_defaults() assigns default labels to unreviewed papers
  - Session store: create / get / delete / expiry
"""
import time

import pytest
from pydantic import ValidationError

from env.environment import PaperReviewEnv
from env.models import Observation, PaperAction, PaperRecord
from env.utils import apply_defaults, sorted_decisions
from server.session import SessionStore


# ── Fixtures (pytest) ─────────────────────────────────────────────────────────

@pytest.fixture
def env():
    return PaperReviewEnv()


@pytest.fixture
def env_t1(env):
    env.reset("task1")
    return env


@pytest.fixture
def env_t2(env):
    env.reset("task2")
    return env


@pytest.fixture
def env_t3(env):
    env.reset("task3")
    return env


@pytest.fixture
def env_t4(env):
    env.reset("task4")
    return env


# ── PaperAction model ─────────────────────────────────────────────────────────

class TestPaperAction:
    def test_valid_review_relevant(self):
        a = PaperAction(action_type="review", paper_id="p001", label="RELEVANT")
        assert a.action_type == "review"
        assert a.paper_id == "p001"
        assert a.label == "RELEVANT"

    def test_valid_submit(self):
        a = PaperAction(action_type="submit")
        assert a.action_type == "submit"
        assert a.paper_id is None

    def test_review_without_paper_id_raises(self):
        with pytest.raises(ValidationError):
            PaperAction(action_type="review")

    def test_invalid_action_type_raises(self):
        with pytest.raises(ValidationError):
            PaperAction(action_type="read", paper_id="p001")

    def test_quality_score_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            PaperAction(action_type="review", paper_id="p001", quality_score=5)

    def test_quality_score_zero_raises(self):
        with pytest.raises(ValidationError):
            PaperAction(action_type="review", paper_id="p001", quality_score=0)

    def test_rank_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            PaperAction(action_type="review", paper_id="p001", rank=6)

    def test_justification_too_long_raises(self):
        with pytest.raises(ValidationError):
            PaperAction(
                action_type="review",
                paper_id="p001",
                justification="x" * 201,
            )

    def test_justification_exactly_200_ok(self):
        a = PaperAction(
            action_type="review",
            paper_id="p001",
            justification="x" * 200,
        )
        assert len(a.justification) == 200

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            PaperAction(action_type="submit", unknown_field="oops")

    def test_serialises_to_dict(self):
        a = PaperAction(action_type="review", paper_id="p001", label="RELEVANT")
        d = a.model_dump(exclude_none=True)
        assert d["action_type"] == "review"
        assert d["paper_id"] == "p001"
        assert "quality_score" not in d


# ── reset() ───────────────────────────────────────────────────────────────────

class TestReset:
    @pytest.mark.parametrize("task_id", ["task1", "task2", "task3", "task4"])
    def test_reset_returns_observation(self, env, task_id):
        obs = env.reset(task_id)
        assert isinstance(obs, Observation)

    @pytest.mark.parametrize("task_id", ["task1", "task2", "task3", "task4"])
    def test_reset_step_zero(self, env, task_id):
        obs = env.reset(task_id)
        assert obs.step == 0

    @pytest.mark.parametrize("task_id,expected_budget", [
        ("task1", 12), ("task2", 14), ("task3", 15), ("task4", 18),
    ])
    def test_reset_budget(self, env, task_id, expected_budget):
        obs = env.reset(task_id)
        assert obs.budget_remaining == expected_budget

    @pytest.mark.parametrize("task_id", ["task1", "task2", "task3", "task4"])
    def test_reset_empty_decisions(self, env, task_id):
        obs = env.reset(task_id)
        assert obs.decisions_so_far == {}

    @pytest.mark.parametrize("task_id", ["task1", "task2", "task3", "task4"])
    def test_reset_not_complete(self, env, task_id):
        obs = env.reset(task_id)
        assert obs.episode_complete is False
        assert obs.final_score is None

    @pytest.mark.parametrize("task_id", ["task1", "task2", "task3", "task4"])
    def test_reset_has_papers(self, env, task_id):
        obs = env.reset(task_id)
        assert len(obs.papers) > 0
        assert all(isinstance(p, PaperRecord) for p in obs.papers)

    @pytest.mark.parametrize("task_id", ["task1", "task2", "task3", "task4"])
    def test_reset_task_description_nonempty(self, env, task_id):
        obs = env.reset(task_id)
        assert isinstance(obs.task_description, str)
        assert len(obs.task_description) > 0

    def test_reset_unknown_task_raises(self, env):
        with pytest.raises(ValueError, match="Unknown task_id"):
            env.reset("task99")

    def test_reset_can_be_called_twice(self, env):
        env.reset("task1")
        obs = env.reset("task2")
        assert obs.task_id == "task2"
        assert obs.step == 0

    def test_reset_no_error_field(self, env):
        obs = env.reset("task1")
        assert obs.error is None

    def test_reset_paper_fields_present(self, env):
        obs = env.reset("task1")
        p = obs.papers[0]
        assert p.id
        assert p.title
        assert p.abstract
        assert p.topic_hint
        assert p.methodology_hint
        assert p.claimed_contribution


# ── step() — budget and state ─────────────────────────────────────────────────

class TestStepBudget:
    def test_step_decrements_budget(self, env_t1):
        obs = env_t1.step(
            PaperAction(action_type="review", paper_id="p001", label="RELEVANT")
        )
        assert obs.budget_remaining == 11

    def test_step_increments_step_counter(self, env_t1):
        obs = env_t1.step(
            PaperAction(action_type="review", paper_id="p001", label="RELEVANT")
        )
        assert obs.step == 1

    def test_multiple_steps_decrement_correctly(self, env_t1):
        first_paper = env_t1.state().papers[0].id
        second_paper = env_t1.state().papers[1].id
        env_t1.step(
            PaperAction(action_type="review", paper_id=first_paper, label="RELEVANT")
        )
        obs = env_t1.step(
            PaperAction(action_type="review", paper_id=second_paper, label="NOT_RELEVANT")
        )
        assert obs.budget_remaining == 10
        assert obs.step == 2

    def test_budget_exhaustion_closes_episode(self, env):
        """Drive budget to zero; episode must auto-close."""
        obs = env.reset("task1")
        paper_ids = [p.id for p in obs.papers]
        labels = ["RELEVANT", "NOT_RELEVANT"] * 10

        while not obs.episode_complete:
            pid = paper_ids[obs.step % len(paper_ids)]
            label = labels[obs.step % len(labels)]
            obs = env.step(
                PaperAction(action_type="review", paper_id=pid, label=label)
            )

        assert obs.episode_complete is True
        assert obs.final_score is not None
        assert isinstance(obs.final_score, float)
        assert 0.0 <= obs.final_score <= 1.0

    def test_step_after_episode_closed_is_noop(self, env_t1):
        env_t1.step(PaperAction(action_type="submit"))
        closed_obs = env_t1.state()
        assert closed_obs.episode_complete

        obs2 = env_t1.step(
            PaperAction(action_type="review", paper_id="p001", label="RELEVANT")
        )
        # Budget must not have changed
        assert obs2.budget_remaining == closed_obs.budget_remaining
        assert obs2.step == closed_obs.step


# ── step() — decision storage ─────────────────────────────────────────────────

class TestStepDecisions:
    def test_valid_action_stored_in_decisions(self, env_t1):
        pid = env_t1.state().papers[0].id
        obs = env_t1.step(
            PaperAction(action_type="review", paper_id=pid, label="RELEVANT")
        )
        assert pid in obs.decisions_so_far
        assert obs.decisions_so_far[pid]["label"] == "RELEVANT"

    def test_overwrite_previous_decision(self, env_t1):
        pid = env_t1.state().papers[0].id
        env_t1.step(
            PaperAction(action_type="review", paper_id=pid, label="RELEVANT")
        )
        obs = env_t1.step(
            PaperAction(action_type="review", paper_id=pid, label="NOT_RELEVANT")
        )
        assert obs.decisions_so_far[pid]["label"] == "NOT_RELEVANT"

    def test_decisions_sorted_by_paper_id(self, env_t1):
        papers = env_t1.state().papers
        # Label all papers
        for p in papers:
            env_t1.step(
                PaperAction(action_type="review", paper_id=p.id, label="RELEVANT")
            )
        obs = env_t1.state()
        keys = list(obs.decisions_so_far.keys())
        assert keys == sorted(keys)

    def test_task2_stores_quality_score(self, env_t2):
        pid = env_t2.state().papers[0].id
        obs = env_t2.step(
            PaperAction(
                action_type="review",
                paper_id=pid,
                label="RELEVANT",
                quality_score=3,
            )
        )
        assert obs.decisions_so_far[pid]["quality_score"] == 3

    def test_task3_defer_stored(self, env_t3):
        pid = env_t3.state().papers[0].id
        obs = env_t3.step(
            PaperAction(action_type="review", paper_id=pid, label="DEFER")
        )
        assert obs.decisions_so_far[pid]["label"] == "DEFER"

    def test_task3_defer_overwritten_by_include(self, env_t3):
        pid = env_t3.state().papers[0].id
        env_t3.step(PaperAction(action_type="review", paper_id=pid, label="DEFER"))
        obs = env_t3.step(
            PaperAction(action_type="review", paper_id=pid, label="INCLUDE")
        )
        assert obs.decisions_so_far[pid]["label"] == "INCLUDE"

    def test_task4_include_stores_rank_and_justification(self, env_t4):
        pid = env_t4.state().papers[0].id
        obs = env_t4.step(
            PaperAction(
                action_type="review",
                paper_id=pid,
                label="INCLUDE",
                rank=1,
                justification="Strong diagnostic ML paper with reproducible results.",
            )
        )
        d = obs.decisions_so_far[pid]
        assert d["label"] == "INCLUDE"
        assert d["rank"] == 1
        assert "justification" in d


# ── step() — invalid action handling ──────────────────────────────────────────

class TestStepInvalidActions:
    def test_unknown_paper_id_sets_error(self, env_t1):
        obs = env_t1.step(
            PaperAction(action_type="review", paper_id="ZZZZ", label="RELEVANT")
        )
        assert obs.error is not None
        assert "ZZZZ" in obs.error

    def test_unknown_paper_id_does_not_update_decisions(self, env_t1):
        obs = env_t1.step(
            PaperAction(action_type="review", paper_id="ZZZZ", label="RELEVANT")
        )
        assert "ZZZZ" not in obs.decisions_so_far

    def test_unknown_paper_id_still_decrements_budget(self, env_t1):
        obs = env_t1.step(
            PaperAction(action_type="review", paper_id="ZZZZ", label="RELEVANT")
        )
        assert obs.budget_remaining == 11

    def test_wrong_label_for_task1_sets_error(self, env_t1):
        pid = env_t1.state().papers[0].id
        obs = env_t1.step(
            PaperAction(action_type="review", paper_id=pid, label="INCLUDE")
        )
        assert obs.error is not None
        assert "INCLUDE" in obs.error

    def test_wrong_label_for_task3_sets_error(self, env_t3):
        pid = env_t3.state().papers[0].id
        obs = env_t3.step(
            PaperAction(action_type="review", paper_id=pid, label="RELEVANT")
        )
        assert obs.error is not None

    def test_task4_include_without_rank_sets_error(self, env_t4):
        pid = env_t4.state().papers[0].id
        obs = env_t4.step(
            PaperAction(
                action_type="review",
                paper_id=pid,
                label="INCLUDE",
                justification="Good paper.",
            )
        )
        assert obs.error is not None
        assert "rank" in obs.error

    def test_task4_include_without_justification_sets_error(self, env_t4):
        pid = env_t4.state().papers[0].id
        obs = env_t4.step(
            PaperAction(
                action_type="review",
                paper_id=pid,
                label="INCLUDE",
                rank=1,
            )
        )
        assert obs.error is not None
        assert "justification" in obs.error

    def test_missing_label_for_task1_sets_error(self, env_t1):
        pid = env_t1.state().papers[0].id
        # Valid Pydantic model but missing required label for this task
        obs = env_t1.step(
            PaperAction(action_type="review", paper_id=pid)
        )
        assert obs.error is not None

    def test_valid_action_clears_previous_error(self, env_t1):
        pid = env_t1.state().papers[0].id
        env_t1.step(
            PaperAction(action_type="review", paper_id="BAD_ID", label="RELEVANT")
        )
        obs = env_t1.step(
            PaperAction(action_type="review", paper_id=pid, label="RELEVANT")
        )
        assert obs.error is None


# ── submit action ─────────────────────────────────────────────────────────────

class TestSubmit:
    def test_submit_closes_episode(self, env_t1):
        obs = env_t1.step(PaperAction(action_type="submit"))
        assert obs.episode_complete is True

    def test_submit_populates_final_score(self, env_t1):
        obs = env_t1.step(PaperAction(action_type="submit"))
        assert obs.final_score is not None
        assert 0.0 <= obs.final_score <= 1.0

    def test_submit_after_some_decisions(self, env_t1):
        pid = env_t1.state().papers[0].id
        env_t1.step(
            PaperAction(action_type="review", paper_id=pid, label="RELEVANT")
        )
        obs = env_t1.step(PaperAction(action_type="submit"))
        assert obs.episode_complete is True
        assert obs.final_score is not None

    def test_submit_on_fresh_episode_returns_zero_score(self, env_t1):
        # No reviews submitted — grader stub returns 0.0
        obs = env_t1.step(PaperAction(action_type="submit"))
        # Phase 2 graders are stubs; score is 0.0
        assert obs.final_score == 0.0

    @pytest.mark.parametrize("task_id", ["task1", "task2", "task3", "task4"])
    def test_submit_works_for_all_tasks(self, env, task_id):
        env.reset(task_id)
        obs = env.step(PaperAction(action_type="submit"))
        assert obs.episode_complete is True
        assert obs.final_score is not None


# ── state() ───────────────────────────────────────────────────────────────────

class TestState:
    def test_state_before_reset_raises(self):
        env = PaperReviewEnv()
        with pytest.raises(RuntimeError):
            env.state()

    def test_state_matches_reset_observation(self, env_t1):
        obs_reset = env_t1.state()
        obs_state = env_t1.state()
        assert obs_reset.step == obs_state.step
        assert obs_reset.budget_remaining == obs_state.budget_remaining
        assert obs_reset.decisions_so_far == obs_state.decisions_so_far

    def test_state_reflects_step(self, env_t1):
        pid = env_t1.state().papers[0].id
        env_t1.step(
            PaperAction(action_type="review", paper_id=pid, label="RELEVANT")
        )
        obs = env_t1.state()
        assert obs.step == 1
        assert obs.budget_remaining == 11
        assert pid in obs.decisions_so_far

    def test_state_does_not_modify_state(self, env_t1):
        before = env_t1.state()
        env_t1.state()
        env_t1.state()
        after = env_t1.state()
        assert before.step == after.step
        assert before.budget_remaining == after.budget_remaining


# ── utils ─────────────────────────────────────────────────────────────────────

class TestUtils:
    def test_sorted_decisions_deterministic(self):
        d = {"p003": {"label": "X"}, "p001": {"label": "Y"}, "p002": {"label": "Z"}}
        result = sorted_decisions(d)
        assert list(result.keys()) == ["p001", "p002", "p003"]

    def test_apply_defaults_fills_missing_papers(self):
        paper_ids = ["p001", "p002", "p003"]
        decisions = {"p001": {"label": "RELEVANT"}}
        result = apply_defaults(paper_ids, decisions, "task1")
        assert "p002" in result
        assert "p003" in result
        assert result["p002"]["label"] == "NOT_RELEVANT"
        assert result["p003"]["label"] == "NOT_RELEVANT"

    def test_apply_defaults_does_not_overwrite_existing(self):
        paper_ids = ["p001", "p002"]
        decisions = {"p001": {"label": "RELEVANT"}}
        result = apply_defaults(paper_ids, decisions, "task1")
        assert result["p001"]["label"] == "RELEVANT"

    def test_apply_defaults_task3_uses_exclude(self):
        result = apply_defaults(["p001"], {}, "task3")
        assert result["p001"]["label"] == "EXCLUDE"

    def test_apply_defaults_task4_uses_exclude(self):
        result = apply_defaults(["p001"], {}, "task4")
        assert result["p001"]["label"] == "EXCLUDE"

    def test_apply_defaults_does_not_mutate_input(self):
        original = {"p001": {"label": "RELEVANT"}}
        apply_defaults(["p001", "p002"], original, "task1")
        assert "p002" not in original


# ── SessionStore ──────────────────────────────────────────────────────────────

class TestSessionStore:
    def test_create_returns_string_id(self):
        store = SessionStore()
        sid = store.create("dummy_env")
        assert isinstance(sid, str)
        assert len(sid) > 0

    def test_get_returns_stored_value(self):
        store = SessionStore()
        sid = store.create("my_env")
        assert store.get(sid) == "my_env"

    def test_get_missing_returns_none(self):
        store = SessionStore()
        assert store.get("nonexistent-id") is None

    def test_delete_removes_session(self):
        store = SessionStore()
        sid = store.create("env")
        store.delete(sid)
        assert store.get(sid) is None

    def test_ttl_expiry(self):
        store = SessionStore(ttl_seconds=0)
        sid = store.create("env_obj")
        time.sleep(0.01)
        # Trigger cleanup via a new create()
        store.create("another")
        assert store.get(sid) is None

    def test_multiple_sessions_independent(self):
        store = SessionStore()
        sid1 = store.create("env1")
        sid2 = store.create("env2")
        assert store.get(sid1) == "env1"
        assert store.get(sid2) == "env2"

    def test_create_unique_ids(self):
        store = SessionStore()
        ids = {store.create(f"env{i}") for i in range(50)}
        assert len(ids) == 50
