from __future__ import annotations

import pytest
from pydantic import ValidationError

from env.models import Observation, PaperAction, PatentRecord


def test_patent_record_valid() -> None:
    rec = PatentRecord(
        id="patent_001",
        title="Test Patent",
        abstract="An abstract.",
        description="A long description.",
        cpc="A01B",
    )
    assert rec.id == "patent_001"


def test_patent_record_allows_missing_optional_fields() -> None:
    rec = PatentRecord(
        id="patent_002",
        abstract="An abstract.",
        description="A long description.",
    )
    assert rec.cpc is None
    assert rec.title is None


def test_paper_action_review_valid() -> None:
    action = PaperAction(
        action_type="review",
        paper_id="patent_001",
        label="RELEVANT",
    )
    assert action.paper_id == "patent_001"


def test_paper_action_submit_valid() -> None:
    action = PaperAction(action_type="submit")
    assert action.action_type == "submit"


def test_paper_action_review_requires_paper_id() -> None:
    with pytest.raises(ValidationError):
        PaperAction(action_type="review", label="RELEVANT")


def test_paper_action_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        PaperAction(action_type="review", paper_id="patent_001", label="RELEVANT", extra_field=1)  # type: ignore[arg-type]


def test_observation_valid() -> None:
    obs = Observation(
        task_id="task1",
        task_description="demo",
        step=0,
        budget_remaining=20,
        query_patent=PatentRecord(
            id="q1",
            abstract="query abstract",
            description="query description",
        ),
        candidate_patents=[],
        decisions_so_far={},
    )
    assert obs.step == 0
