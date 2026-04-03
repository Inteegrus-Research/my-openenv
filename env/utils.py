"""
env/utils.py
Shared constants and pure helper functions.
No side effects. No external calls. No randomness.
"""
from typing import Any, Dict, List

# ── Per-task step budgets ─────────────────────────────────────────────────────
TASK_BUDGETS: Dict[str, int] = {
    "task1": 12,
    "task2": 14,
    "task3": 15,
    "task4": 18,
}

# ── Reference paper counts (fixture determines actual; not enforced at runtime) ──
TASK_PAPER_COUNTS: Dict[str, int] = {
    "task1": 10,
    "task2": 10,
    "task3": 15,
    "task4": 20,
}

# ── Valid label values per task ───────────────────────────────────────────────
VALID_LABELS: Dict[str, frozenset] = {
    "task1": frozenset({"RELEVANT", "NOT_RELEVANT"}),
    "task2": frozenset({"RELEVANT", "NOT_RELEVANT"}),
    "task3": frozenset({"INCLUDE", "EXCLUDE", "DEFER"}),
    "task4": frozenset({"INCLUDE", "EXCLUDE"}),
}

# ── Default label for papers with no decision at episode end ──────────────────
DEFAULT_LABELS: Dict[str, str] = {
    "task1": "NOT_RELEVANT",
    "task2": "NOT_RELEVANT",
    "task3": "EXCLUDE",
    "task4": "EXCLUDE",
}

# ── Recognised task IDs ───────────────────────────────────────────────────────
TASK_IDS: frozenset = frozenset(TASK_BUDGETS.keys())


# ── Helpers ───────────────────────────────────────────────────────────────────

def sorted_decisions(decisions: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a new dict sorted by paper_id (lexicographic).
    Guarantees deterministic JSON output regardless of insertion order.
    """
    return dict(sorted(decisions.items()))


def get_default_label(task_id: str) -> str:
    """Return the label applied to papers not reviewed by episode end."""
    return DEFAULT_LABELS[task_id]


def apply_defaults(
    paper_ids: List[str],
    decisions: Dict[str, Any],
    task_id: str,
) -> Dict[str, Any]:
    """
    Return a new decisions dict that includes every paper_id.
    Papers absent from decisions receive a minimal default-label action.
    Paper IDs are processed in sorted order for determinism.
    Does not mutate the input dict.
    """
    default_label = get_default_label(task_id)
    result = dict(decisions)
    for pid in sorted(paper_ids):
        if pid not in result:
            result[pid] = {
                "action_type": "review",
                "paper_id": pid,
                "label": default_label,
            }
    return result
