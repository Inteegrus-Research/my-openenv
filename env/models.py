"""
env/models.py
All Pydantic models for OpenEnv-PaperBench.

Three models only:
  PaperRecord  — one paper in the batch (always fully visible)
  PaperAction  — unified action schema for all four tasks
  Observation  — flat observation returned by reset() and step()
"""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PaperRecord(BaseModel):
    """
    One paper in the screening batch.
    All fields are always present and always visible from step 0.
    Extra fields in the fixture JSON are silently ignored.
    """
    model_config = ConfigDict(extra="ignore")

    id: str
    title: str
    abstract: str
    topic_hint: str
    methodology_hint: str
    claimed_contribution: str


class PaperAction(BaseModel):
    """
    Unified action schema for all four tasks.

    action_type:
        "review"  — act on a specific paper (paper_id required)
        "submit"  — close the episode immediately

    Task-specific field requirements are enforced by the task's validate_action()
    method, not by this model.  The model only enforces structural constraints:
        - paper_id required for "review"
        - quality_score in [1, 4] if provided
        - rank in [1, 5] if provided
        - justification ≤ 200 chars if provided
    """
    model_config = ConfigDict(extra="forbid")

    action_type: Literal["review", "submit"]
    paper_id: Optional[str] = None
    label: Optional[str] = None
    quality_score: Optional[int] = Field(default=None, ge=1, le=4)
    rank: Optional[int] = Field(default=None, ge=1, le=5)
    justification: Optional[str] = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def _review_requires_paper_id(self) -> "PaperAction":
        if self.action_type == "review" and self.paper_id is None:
            raise ValueError(
                "paper_id is required when action_type is 'review'."
            )
        return self


class Observation(BaseModel):
    """
    Flat observation returned after every reset() and step() call.

    - papers is the full batch, always complete.
    - decisions_so_far maps paper_id -> the agent's most recent action dict.
    - error is None when the last action was valid; a message string otherwise.
    - final_score and episode_complete are only meaningful when episode_complete=True.
    """
    model_config = ConfigDict(extra="forbid")

    task_id: str
    task_description: str
    step: int
    budget_remaining: int
    papers: List[PaperRecord]
    decisions_so_far: Dict[str, Any]   # paper_id -> dict of stored action fields
    episode_complete: bool = False
    final_score: Optional[float] = None
    error: Optional[str] = None
