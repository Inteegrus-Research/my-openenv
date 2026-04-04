"""
graders/graders.py
Deterministic grading functions for all four tasks.

Each function is a pure function:
    grade_taskN(completed_decisions, fixture, **kwargs) -> float in [0.0, 1.0]

completed_decisions:
    Dict[paper_id -> action_dict] with every paper present (defaults applied).

fixture:
    The loaded fixture dict, including fixture["ground_truth"].

All functions are deterministic, side-effect-free, and wrapped in try/except.
"""
import math
from typing import Any, Dict, Optional


# ── Helpers ───────────────────────────────────────────────────────────────────

def _f1_binary(y_true, y_pred) -> float:
    """Binary F1, positive class = 1. Returns 0.0 on no positives."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _ndcg_at_k(agent_ranks: Dict[str, int], rq_scores: Dict[str, float], k: int = 5) -> float:
    """
    nDCG@k. agent_ranks: paper_id -> 1-indexed rank. rq_scores: paper_id -> float gain.
    Duplicate ranks: keep highest gain at that position.
    """
    ideal = sorted(rq_scores.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    if idcg <= 0.0:
        return 0.0

    pos_gain: Dict[int, float] = {}
    for pid, rank in agent_ranks.items():
        if 1 <= rank <= k:
            gain = rq_scores.get(pid, 0.0)
            pos  = rank - 1
            if pos not in pos_gain or gain > pos_gain[pos]:
                pos_gain[pos] = gain

    dcg = sum(pos_gain.get(i, 0.0) / math.log2(i + 2) for i in range(k))
    return float(min(1.0, dcg / idcg))


def _justification_valid(text: Any, vocab: frozenset) -> bool:
    """Non-empty, <=200 chars, contains at least one vocab alpha-token."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return False
    if len(text) > 200:
        return False
    tokens = {w.strip(".,;:!?()[]{}\"'").lower() for w in text.split()}
    alpha_tokens = {t for t in tokens if t.isalpha()}
    return bool(alpha_tokens & vocab)


# ── Task 1 ────────────────────────────────────────────────────────────────────

def grade_task1(completed_decisions: Dict[str, Any], fixture: dict, **_kw) -> float:
    """Binary relevance F1 (RELEVANT = positive class)."""
    try:
        gt = fixture["ground_truth"]["labels"]
        paper_ids = sorted(gt.keys())
        y_true, y_pred = [], []
        for pid in paper_ids:
            y_true.append(1 if gt[pid] == "RELEVANT" else 0)
            pred_label = completed_decisions.get(pid, {}).get("label", "NOT_RELEVANT")
            y_pred.append(1 if pred_label == "RELEVANT" else 0)
        return float(max(0.0, min(1.0, _f1_binary(y_true, y_pred))))
    except Exception:
        return 0.0


# ── Task 2 ────────────────────────────────────────────────────────────────────

def grade_task2(completed_decisions: Dict[str, Any], fixture: dict, **_kw) -> float:
    """0.6 * relevance_F1 + 0.4 * quality_accuracy (1 - MAE/3)."""
    try:
        gt_labels  = fixture["ground_truth"]["labels"]
        gt_quality = fixture["ground_truth"]["quality_scores"]
        paper_ids  = sorted(gt_labels.keys())

        y_true_rel, y_pred_rel, abs_errors = [], [], []
        for pid in paper_ids:
            y_true_rel.append(1 if gt_labels[pid] == "RELEVANT" else 0)
            pred   = completed_decisions.get(pid, {})
            y_pred_rel.append(1 if pred.get("label", "NOT_RELEVANT") == "RELEVANT" else 0)

            true_q = int(gt_quality[pid])
            pred_q = max(1, min(4, int(pred.get("quality_score", 1))))
            abs_errors.append(abs(true_q - pred_q))

        rel_f1   = _f1_binary(y_true_rel, y_pred_rel)
        mae      = sum(abs_errors) / len(abs_errors) if abs_errors else 3.0
        qual_acc = max(0.0, 1.0 - mae / 3.0)
        return float(max(0.0, min(1.0, 0.6 * rel_f1 + 0.4 * qual_acc)))
    except Exception:
        return 0.0


# ── Task 3 ────────────────────────────────────────────────────────────────────

def grade_task3(
    completed_decisions: Dict[str, Any],
    fixture: dict,
    steps_used: Optional[int] = None,
    **_kw,
) -> float:
    """0.85 * F1 + 0.15 * budget_efficiency. DEFER counts as EXCLUDE."""
    try:
        gt        = fixture["ground_truth"]["labels"]
        budget    = int(fixture.get("budget", 15))
        paper_ids = sorted(gt.keys())

        y_true, y_pred = [], []
        for pid in paper_ids:
            y_true.append(1 if gt[pid] == "INCLUDE" else 0)
            label = completed_decisions.get(pid, {}).get("label", "EXCLUDE")
            if label == "DEFER":
                label = "EXCLUDE"
            y_pred.append(1 if label == "INCLUDE" else 0)

        f1 = _f1_binary(y_true, y_pred)

        efficiency = 0.0
        if steps_used is not None and budget > 0:
            efficiency = max(0.0, min(1.0, (budget - steps_used) / budget))

        return float(max(0.0, min(1.0, 0.85 * f1 + 0.15 * efficiency)))
    except Exception:
        return 0.0


# ── Task 4 ────────────────────────────────────────────────────────────────────

def grade_task4(completed_decisions: Dict[str, Any], fixture: dict, **_kw) -> float:
    """0.50 * nDCG@5 + 0.35 * F1 + 0.15 * justification_validity."""
    try:
        gt_labels = fixture["ground_truth"]["labels"]
        rq_scores = {k: float(v) for k, v in
                     fixture["ground_truth"]["relevance_quality_scores"].items()}
        vocab     = frozenset(
            w.strip().lower()
            for w in fixture["ground_truth"]["vocabulary_list"]
        )
        paper_ids = sorted(gt_labels.keys())

        # Binary F1
        y_true, y_pred = [], []
        for pid in paper_ids:
            y_true.append(1 if gt_labels[pid] == "INCLUDE" else 0)
            y_pred.append(
                1 if completed_decisions.get(pid, {}).get("label", "EXCLUDE") == "INCLUDE"
                else 0
            )
        f1 = _f1_binary(y_true, y_pred)

        # nDCG@5
        agent_ranks: Dict[str, int] = {}
        for pid, d in completed_decisions.items():
            if d.get("label") == "INCLUDE":
                r = d.get("rank")
                if r is not None and 1 <= int(r) <= 5:
                    agent_ranks[pid] = int(r)
        ndcg = _ndcg_at_k(agent_ranks, rq_scores, k=5)

        # Justification validity
        include_pids = [p for p, d in completed_decisions.items()
                        if d.get("label") == "INCLUDE"]
        if not include_pids:
            just_score = 0.0
        else:
            valid = sum(
                1 for pid in include_pids
                if _justification_valid(
                    completed_decisions[pid].get("justification"), vocab
                )
            )
            just_score = valid / len(include_pids)

        score = 0.50 * ndcg + 0.35 * f1 + 0.15 * just_score
        return float(max(0.0, min(1.0, score)))
    except Exception:
        return 0.0
