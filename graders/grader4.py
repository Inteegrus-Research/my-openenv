from __future__ import annotations

from typing import Any, Dict

from graders.grader1 import _clamp_open, _f1, _justification_ok, _ndcg_at_k, _paper_ids, _top_k_positive_ids


def grade_task4(completed_decisions: Dict[str, Any], fixture: dict, **kwargs) -> float:
    try:
        ids = _paper_ids(fixture)
        positives = _top_k_positive_ids(fixture, k=5)
        y_true = [1 if pid in positives else 0 for pid in ids]
        y_pred = []
        for pid in ids:
            label = str(completed_decisions.get(pid, {}).get("label", "EXCLUDE")).upper()
            y_pred.append(1 if label == "INCLUDE" else 0)

        f1 = _f1(y_true, y_pred)

        gains = {pid: float(score) for pid, score in fixture["ground_truth"]["relevance_scores"].items()}
        ranking = []
        ranked_items = []
        for pid, action in completed_decisions.items():
            if str(action.get("label", "")).upper() != "INCLUDE":
                continue
            rank = action.get("rank", None)
            if rank is None:
                continue
            try:
                rank = int(rank)
            except Exception:
                continue
            if 1 <= rank <= 5:
                ranked_items.append((rank, pid))
        ranked_items.sort(key=lambda x: x[0])
        ranking = [pid for _, pid in ranked_items]
        ndcg = _ndcg_at_k(gains, ranking, k=5)

        vocab = set()
        q = fixture["query_patent"]
        for field in ("abstract", "description", "cpc"):
            vocab.update([t for t in __import__("re").findall(r"[a-zA-Z0-9]+", str(q.get(field, "")).lower()) if len(t) >= 5])

        include_ids = [pid for pid, action in completed_decisions.items() if str(action.get("label", "")).upper() == "INCLUDE"]
        if include_ids:
            valid = sum(1 for pid in include_ids if _justification_ok(completed_decisions[pid].get("justification"), vocab))
            just = valid / len(include_ids)
        else:
            just = 0.0

        return _clamp_open(0.50 * ndcg + 0.35 * f1 + 0.15 * just)
    except Exception:
        return 0.01