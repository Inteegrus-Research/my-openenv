from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PatentRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    title: Optional[str] = None
    abstract: str
    description: str
    cpc: Optional[str] = None


class PaperAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: Literal["review", "submit"]
    paper_id: Optional[str] = None
    label: Optional[str] = None
    quality_score: Optional[int] = Field(default=None, ge=1, le=4)
    rank: Optional[int] = Field(default=None, ge=1, le=5)
    justification: Optional[str] = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def _paper_id_required_for_review(self) -> "PaperAction":
        if self.action_type == "review" and not self.paper_id:
            raise ValueError("paper_id is required when action_type='review'")
        return self


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    task_description: str
    step: int
    budget_remaining: int
    query_patent: PatentRecord
    candidate_patents: List[PatentRecord]
    decisions_so_far: Dict[str, Dict[str, Any]]
    episode_complete: bool = False
    final_score: Optional[float] = None
    error: Optional[str] = None


class Reward(BaseModel):
    value: float