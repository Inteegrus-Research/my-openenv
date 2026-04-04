"""
scripts/validate_fixtures.py
Dev-only fixture validator.

Checks:
1. Every fixture file loads as valid JSON with required keys.
2. Every grader runs without crashing on the fixture.
3. Three degenerate baselines (all-positive, all-negative, random) produce
   non-trivially different scores — no constant-score grader.
4. All returned scores are in [0.0, 1.0].
5. No grader crashes on an empty episode.

Run from the repo root:
    python scripts/validate_fixtures.py

Exits with code 0 on success, code 1 on any failure.
"""
import json
import random
import sys
from pathlib import Path

# --- ensure project root is on sys.path ---------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from graders.graders import grade_task1, grade_task2, grade_task3, grade_task4
from env.utils import TASK_BUDGETS

FIXTURES_DIR = ROOT / "fixtures"
GRADERS = {
    "task1": grade_task1,
    "task2": grade_task2,
    "task3": grade_task3,
    "task4": grade_task4,
}
REQUIRED_KEYS = {"task_description", "papers", "ground_truth"}


def load_fixture(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def all_positive_decisions(fixture: dict, task_id: str) -> dict:
    """Every paper gets the positive label for this task."""
    label = "RELEVANT" if task_id in ("task1", "task2") else "INCLUDE"
    return {
        p["id"]: {
            "action_type": "review",
            "paper_id": p["id"],
            "label": label,
            "quality_score": 4 if task_id == "task2" else None,
            "rank": None,
            "justification": None,
        }
        for p in fixture["papers"]
    }


def all_negative_decisions(fixture: dict, task_id: str) -> dict:
    """Every paper gets the negative label for this task."""
    label = "NOT_RELEVANT" if task_id in ("task1", "task2") else "EXCLUDE"
    return {
        p["id"]: {
            "action_type": "review",
            "paper_id": p["id"],
            "label": label,
            "quality_score": 1 if task_id == "task2" else None,
        }
        for p in fixture["papers"]
    }


def random_decisions(fixture: dict, task_id: str, seed: int = 42) -> dict:
    """Random label per paper using fixed seed."""
    rng = random.Random(seed)
    if task_id == "task1":
        choices = ["RELEVANT", "NOT_RELEVANT"]
    elif task_id == "task2":
        choices = ["RELEVANT", "NOT_RELEVANT"]
    elif task_id == "task3":
        choices = ["INCLUDE", "EXCLUDE", "DEFER"]
    else:
        choices = ["INCLUDE", "EXCLUDE"]

    decisions = {}
    rank_counter = 1
    for p in fixture["papers"]:
        label = rng.choice(choices)
        d: dict = {"action_type": "review", "paper_id": p["id"], "label": label}
        if task_id == "task2":
            d["quality_score"] = rng.randint(1, 4)
        if task_id == "task4" and label == "INCLUDE" and rank_counter <= 5:
            d["rank"] = rank_counter
            d["justification"] = "Relevant clinical ML paper with evaluation."
            rank_counter += 1
        decisions[p["id"]] = d
    return decisions


def perfect_decisions(fixture: dict, task_id: str) -> dict:
    """Ground-truth-perfect decisions."""
    gt = fixture["ground_truth"]
    decisions = {}
    rank_counter = 1

    if task_id in ("task1", "task2"):
        for pid, label in gt["labels"].items():
            d: dict = {"action_type": "review", "paper_id": pid, "label": label}
            if task_id == "task2":
                d["quality_score"] = gt["quality_scores"][pid]
            decisions[pid] = d

    elif task_id == "task3":
        for pid, label in gt["labels"].items():
            decisions[pid] = {"action_type": "review", "paper_id": pid, "label": label}

    else:  # task4
        ranked = gt.get("ranked_order", [])
        vocab = gt.get("vocabulary_list", ["model", "learning"])
        for pid, label in gt["labels"].items():
            d_: dict = {"action_type": "review", "paper_id": pid, "label": label}
            if label == "INCLUDE" and rank_counter <= 5:
                d_["rank"] = rank_counter
                # Use a word from vocab in the justification
                word = vocab[0] if vocab else "learning"
                d_["justification"] = f"High-quality {word} paper with rigorous evaluation and baselines."[:200]
                rank_counter += 1
            decisions[pid] = d_
    return decisions


def run_checks():
    errors = []
    print("=" * 60)
    print("OpenEnv-PaperBench fixture validator")
    print("=" * 60)

    for task_id in ["task1", "task2", "task3", "task4"]:
        task_dir = FIXTURES_DIR / task_id
        json_files = sorted(task_dir.glob("*.json"))

        if not json_files:
            errors.append(f"  [FAIL] {task_id}: no fixture files found")
            continue

        for fpath in json_files:
            label = f"{task_id}/{fpath.name}"
            print(f"\n--- {label} ---")

            # 1. Load
            try:
                fixture = load_fixture(fpath)
            except Exception as e:
                errors.append(f"  [FAIL] {label}: JSON load error: {e}")
                continue

            # 2. Schema
            missing = REQUIRED_KEYS - set(fixture.keys())
            if missing:
                errors.append(f"  [FAIL] {label}: missing keys {missing}")
                continue
            papers = fixture["papers"]
            if not papers:
                errors.append(f"  [FAIL] {label}: empty papers list")
                continue
            print(f"  schema ok ({len(papers)} papers)")

            grader_fn = GRADERS[task_id]
            budget = TASK_BUDGETS.get(task_id, 15)

            # 3. Empty episode
            try:
                score = grader_fn({}, fixture, steps_used=budget)
                assert isinstance(score, float), "not a float"
                assert 0.0 <= score <= 1.0, f"out of range: {score}"
                print(f"  empty episode -> {score:.4f}  ok")
            except Exception as e:
                errors.append(f"  [FAIL] {label}: empty episode crash: {e}")

            # 4. Baseline scores
            baselines = {
                "all_positive": all_positive_decisions(fixture, task_id),
                "all_negative": all_negative_decisions(fixture, task_id),
                "random":       random_decisions(fixture, task_id),
                "perfect":      perfect_decisions(fixture, task_id),
            }
            scores = {}
            for name, decisions in baselines.items():
                try:
                    s = grader_fn(decisions, fixture, steps_used=budget)
                    assert isinstance(s, float)
                    assert 0.0 <= s <= 1.0, f"out of range: {s}"
                    scores[name] = s
                    print(f"  {name:14s} -> {s:.4f}")
                except Exception as e:
                    errors.append(f"  [FAIL] {label}: {name} crashed: {e}")

            # 5. Sanity: perfect must beat all-positive and all-negative
            if "perfect" in scores and "all_positive" in scores:
                if scores["perfect"] < scores["all_positive"] and task_id != "task3":
                    errors.append(
                        f"  [WARN] {label}: perfect ({scores['perfect']:.4f}) "
                        f"< all_positive ({scores['all_positive']:.4f})"
                    )
            if "perfect" in scores and "all_negative" in scores:
                if scores["perfect"] < 0.3:
                    errors.append(
                        f"  [WARN] {label}: perfect score is very low "
                        f"({scores['perfect']:.4f}) — check ground truth"
                    )

            # 6. Non-constant: all_positive != all_negative
            if "all_positive" in scores and "all_negative" in scores:
                if abs(scores["all_positive"] - scores["all_negative"]) < 0.05:
                    errors.append(
                        f"  [FAIL] {label}: constant-score grader — "
                        f"all_positive={scores['all_positive']:.4f} "
                        f"all_negative={scores['all_negative']:.4f}"
                    )
                else:
                    print(f"  non-constant grader: ok")

    print("\n" + "=" * 60)
    if errors:
        print(f"VALIDATION FAILED — {len(errors)} issue(s):")
        for e in errors:
            print(e)
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    run_checks()
