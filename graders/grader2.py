from __future__ import annotations

from typing import Any, Dict

from graders.grader1 import _clamp_open, _f1, _paper_ids, _pred_binary_map, _quantize_score, _truth_binary_map


def grade_task2(completed_decisions: Dict[str, Any], fixture: dict, **kwargs) -> float:
    try:
        ids = _paper_ids(fixture)
        y_true_map = _truth_binary_map(fixture)
        y_pred_map = _pred_binary_map(completed_decisions, "RELEVANT")
        y_true = [y_true_map.get(pid, 0) for pid in ids]
        y_pred = [y_pred_map.get(pid, 0) for pid in ids]

        rel_f1 = _f1(y_true, y_pred)

        gt_scores = fixture["ground_truth"]["relevance_scores"]
        true_q = [_quantize_score(float(gt_scores.get(pid, 0.0))) for pid in ids]
        pred_q = [int(completed_decisions.get(pid, {}).get("quality_score", 1) or 1) for pid in ids]
        pred_q = [max(1, min(4, q)) for q in pred_q]
        mae = sum(abs(a - b) for a, b in zip(true_q, pred_q)) / max(1, len(ids))
        qacc = max(0.0, 1.0 - (mae / 3.0))

        return _clamp_open(0.6 * rel_f1 + 0.4 * qacc)
    except Exception:
        return 0.01