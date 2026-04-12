#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"

TASK_BUDGETS = {"task1": 20, "task2": 24, "task3": 28, "task4": 32}

TASK_DESCRIPTIONS = {
    "task1": "Screen candidate patents for relevance to the query patent. Mark the strongest prior-art candidates as RELEVANT and the rest as NOT_RELEVANT.",
    "task2": "Rank candidate patents by novelty risk and provide a quality score from 1 to 4 for each review decision.",
    "task3": "Decide whether each candidate should be INCLUDED, EXCLUDED, or DEFERRED. DEFER counts as EXCLUDE at grading time.",
    "task4": "Produce a ranked top-5 prior-art shortlist with justification strings for INCLUDE decisions. This is the hardest task.",
}

random.seed(42)


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a JSON object")
    return data


def _check_fixture(task_id: str, fixture: dict[str, Any]) -> None:
    # Required top-level keys
    required = {"task_description", "budget", "query_patent", "candidate_patents", "ground_truth"}
    missing = required - set(fixture.keys())
    assert not missing, f"Missing keys: {missing}"

    # Check task_description matches expected
    expected_desc = TASK_DESCRIPTIONS[task_id]
    assert fixture["task_description"] == expected_desc, (
        f"task_description mismatch:\n  got: {fixture['task_description']}\n  expected: {expected_desc}"
    )

    # Check budget field
    assert fixture["budget"] == TASK_BUDGETS[task_id], (
        f"Budget mismatch: got {fixture['budget']}, expected {TASK_BUDGETS[task_id]}"
    )

    # Candidates
    candidates = fixture["candidate_patents"]
    assert isinstance(candidates, list) and candidates, "candidate_patents empty or not list"
    for p in candidates:
        assert "id" in p, "Candidate missing id"
        assert "abstract" in p, "Candidate missing abstract"
        assert "cpc" in p, "Candidate missing cpc"

    # Ground truth
    gt = fixture["ground_truth"]
    assert isinstance(gt, dict), "ground_truth must be dict"
    assert "ranking" in gt and isinstance(gt["ranking"], list), "ranking missing or not list"
    assert "relevance_scores" in gt and isinstance(gt["relevance_scores"], dict), "relevance_scores missing or not dict"
    assert "novelty_score" in gt, "novelty_score missing"

    # All candidate IDs must appear in relevance_scores
    candidate_ids = {p["id"] for p in candidates}
    score_ids = set(gt["relevance_scores"].keys())
    assert candidate_ids == score_ids, f"Mismatch between candidates and scores: {candidate_ids ^ score_ids}"

    # Scores in [0.01, 0.99]
    for pid, score in gt["relevance_scores"].items():
        assert 0.0 <= score <= 1.0, f"Score out of range for {pid}: {score}"

    # Ranking must contain all candidate IDs
    assert set(gt["ranking"]) == candidate_ids, "Ranking does not contain all candidate IDs"

    # Novelty score in [0.1, 0.9]
    assert 0.1 <= gt["novelty_score"] <= 0.9, f"novelty_score out of range: {gt['novelty_score']}"


def main() -> None:
    print("=" * 60)
    print("PriorArtBench fixture validator (post‑generation)")
    print("=" * 60)

    all_passed = True
    for task_dir in sorted(FIXTURES.glob("task*")):
        task_id = task_dir.name
        for path in sorted(task_dir.glob("instance_*.json")):
            try:
                fixture = _load(path)
                _check_fixture(task_id, fixture)
                print(f"✓ {task_id}/{path.name}")
            except Exception as e:
                print(f"✗ {task_id}/{path.name} – {e}")
                all_passed = False

    if all_passed:
        print("\n" + "=" * 60)
        print("ALL FIXTURES VALIDATED SUCCESSFULLY")
    else:
        print("\n" + "=" * 60)
        print("FIXTURE VALIDATION FAILED – see errors above")
        exit(1)


if __name__ == "__main__":
    main()