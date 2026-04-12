from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple


def _clamp_open(score: float) -> float:
    return max(0.01, min(0.99, float(score)))


def _paper_ids(fixture: dict) -> List[str]:
    return [p["id"] for p in fixture["candidate_patents"]]


def _top_k_positive_ids(fixture: dict, k: int | None = None) -> Set[str]:
    scores = fixture["ground_truth"]["relevance_scores"]
    ranked = sorted(scores.items(), key=lambda x: (x[1], x[0]), reverse=True)
    if k is None:
        k = max(10, len(ranked) // 5)
    return {pid for pid, _ in ranked[:k]}


def _truth_binary_map(fixture: dict, k: int | None = None) -> Dict[str, int]:
    positives = _top_k_positive_ids(fixture, k=k)
    return {pid: int(pid in positives) for pid in _paper_ids(fixture)}


def _pred_binary_map(completed_decisions: Dict[str, Any], positive_label: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for pid, action in completed_decisions.items():
        label = str(action.get("label", "")).upper()
        out[pid] = int(label == positive_label)
    return out


def _f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return 0.0 if (prec + rec) == 0.0 else (2 * prec * rec / (prec + rec))


def _quantize_score(score: float) -> int:
    if score < 0.25:
        return 1
    if score < 0.5:
        return 2
    if score < 0.75:
        return 3
    return 4


def _ndcg_at_k(gains_by_id: Dict[str, float], ranking: Sequence[str], k: int = 5) -> float:
    ideal = sorted(gains_by_id.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    if idcg <= 0:
        return 0.0
    dcg = 0.0
    for i, pid in enumerate(ranking[:k]):
        dcg += gains_by_id.get(pid, 0.0) / math.log2(i + 2)
    return dcg / idcg


def _justification_ok(text: Any, vocab: Set[str]) -> bool:
    if not isinstance(text, str) or not text.strip() or len(text) > 200:
        return False
    toks = set(re.findall(r"[a-zA-Z0-9]+", text.lower()))
    return bool(toks & vocab)


def grade_task1(completed_decisions: Dict[str, Any], fixture: dict, **kwargs) -> float:
    try:
        y_true_map = _truth_binary_map(fixture)
        y_pred_map = _pred_binary_map(completed_decisions, "RELEVANT")
        ids = _paper_ids(fixture)
        y_true = [y_true_map.get(pid, 0) for pid in ids]
        y_pred = [y_pred_map.get(pid, 0) for pid in ids]
        return _clamp_open(_f1(y_true, y_pred))
    except Exception:
        return 0.01