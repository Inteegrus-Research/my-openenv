"""
tests/test_graders.py
Phase 3 grader test suite.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from graders.graders import (
    _f1_binary,
    _justification_valid,
    _ndcg_at_k,
    grade_task1,
    grade_task2,
    grade_task3,
    grade_task4,
)


# ── Fixture helpers ──────────────────────────────────────────────────────────

def load(task: str, inst: str = "instance_001") -> dict:
    return json.load(open(ROOT / "fixtures" / task / f"{inst}.json"))

def paper_ids(f):
    return [p["id"] for p in f["papers"]]

def all_positive(f, task):
    label = "RELEVANT" if task in ("task1","task2") else "INCLUDE"
    rank_c = 1
    d = {}
    for p in f["papers"]:
        e: dict = {"label": label}
        if task == "task2": e["quality_score"] = 4
        if task == "task4" and rank_c <= 5:
            e["rank"] = rank_c
            e["justification"] = "Strong clinical diagnosis model with trained evaluation."
            rank_c += 1
        d[p["id"]] = e
    return d

def all_negative(f, task):
    label = "NOT_RELEVANT" if task in ("task1","task2") else "EXCLUDE"
    d = {}
    for p in f["papers"]:
        e: dict = {"label": label}
        if task == "task2": e["quality_score"] = 1
        d[p["id"]] = e
    return d

def perfect(f, task):
    gt = f["ground_truth"]
    decisions = {}
    rank_c = 1
    if task in ("task1","task2"):
        for pid, lbl in gt["labels"].items():
            d: dict = {"label": lbl}
            if task == "task2": d["quality_score"] = gt["quality_scores"][pid]
            decisions[pid] = d
    elif task == "task3":
        for pid, lbl in gt["labels"].items():
            decisions[pid] = {"label": lbl}
    else:
        vocab = gt["vocabulary_list"]
        word = vocab[0] if vocab else "model"
        for pid, lbl in gt["labels"].items():
            d_: dict = {"label": lbl}
            if lbl == "INCLUDE" and rank_c <= 5:
                d_["rank"] = rank_c
                d_["justification"] = f"Automated clinical {word} trained network ablation evaluation."[:200]
                rank_c += 1
            decisions[pid] = d_
    return decisions


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestF1Binary:
    def test_perfect(self):
        assert _f1_binary([1,1,0,0],[1,1,0,0]) == pytest.approx(1.0)
    def test_all_wrong(self):
        assert _f1_binary([1,1,0,0],[0,0,1,1]) == pytest.approx(0.0)
    def test_all_positive_pred_half_true(self):
        assert _f1_binary([1,1,0,0],[1,1,1,1]) == pytest.approx(2/3, rel=1e-5)
    def test_all_negative_pred(self):
        assert _f1_binary([1,1,0,0],[0,0,0,0]) == pytest.approx(0.0)
    def test_no_true_positives(self):
        assert _f1_binary([0,0,0,0],[1,1,0,0]) == pytest.approx(0.0)
    def test_single_correct(self):
        assert _f1_binary([1,0],[1,0]) == pytest.approx(1.0)


class TestNdcgAtK:
    def test_perfect(self):
        assert _ndcg_at_k({"p1":1,"p2":2,"p3":3},{"p1":1.0,"p2":0.8,"p3":0.6,"p4":0.1},k=3) == pytest.approx(1.0)
    def test_empty(self):
        assert _ndcg_at_k({},{"p1":1.0},k=5) == pytest.approx(0.0)
    def test_wrong_order_lower(self):
        rq = {"p1":1.0,"p2":0.1}
        assert _ndcg_at_k({"p1":1,"p2":2},rq,k=2) > _ndcg_at_k({"p1":2,"p2":1},rq,k=2)
    def test_rank_beyond_k_ignored(self):
        assert _ndcg_at_k({"p1":6},{"p1":1.0},k=5) == pytest.approx(0.0)
    def test_partial_positive(self):
        s = _ndcg_at_k({"p1":1},{"p1":1.0,"p2":0.9},k=3)
        assert 0.0 < s < 1.0


class TestJustificationValid:
    V = frozenset({"model","trained","clinical"})
    def test_valid(self):
        assert _justification_valid("This model was trained for clinical diagnosis.", self.V)
    def test_empty(self):
        assert not _justification_valid("", self.V)
    def test_whitespace(self):
        assert not _justification_valid("   ", self.V)
    def test_too_long(self):
        assert not _justification_valid("model " * 35, self.V)
    def test_exactly_200_valid(self):
        text = "model " + "x" * 194
        assert len(text) == 200
        assert _justification_valid(text, self.V)
    def test_no_overlap(self):
        assert not _justification_valid("This paper is interesting.", self.V)
    def test_none(self):
        assert not _justification_valid(None, self.V)
    def test_punct_stripped(self):
        assert _justification_valid("model.", self.V)


# ══════════════════════════════════════════════════════════════════════════════
# Task 1
# ══════════════════════════════════════════════════════════════════════════════

class TestGradeTask1:
    def setup_method(self):
        self.f = load("task1")
        self.pids = paper_ids(self.f)

    def test_returns_float(self):
        assert isinstance(grade_task1({}, self.f), float)
    def test_empty_zero(self):
        assert grade_task1({}, self.f) == pytest.approx(0.0)
    def test_perfect_one(self):
        assert grade_task1(perfect(self.f,"task1"), self.f) == pytest.approx(1.0)
    def test_all_pos_lt_perfect(self):
        assert grade_task1(all_positive(self.f,"task1"),self.f) < grade_task1(perfect(self.f,"task1"),self.f)
    def test_all_neg_zero(self):
        assert grade_task1(all_negative(self.f,"task1"),self.f) == pytest.approx(0.0)
    def test_pos_neq_neg(self):
        assert abs(grade_task1(all_positive(self.f,"task1"),self.f) -
                   grade_task1(all_negative(self.f,"task1"),self.f)) > 0.05
    def test_bounds(self):
        for d in [all_positive(self.f,"task1"), all_negative(self.f,"task1"), {}]:
            assert 0.0 <= grade_task1(d,self.f) <= 1.0
    def test_unknown_label_treated_as_neg(self):
        d = {pid:{"label":"BANANA"} for pid in self.pids}
        assert grade_task1(d,self.f) == pytest.approx(0.0)
    def test_missing_label_key(self):
        d = {pid:{} for pid in self.pids}
        s = grade_task1(d,self.f)
        assert 0.0 <= s <= 1.0
    def test_score_changes_with_decisions(self):
        d1 = {pid:{"label":"RELEVANT"} for pid in self.pids}
        d2 = {pid:{"label":"NOT_RELEVANT"} for pid in self.pids}
        assert grade_task1(d1,self.f) != grade_task1(d2,self.f)
    def test_deterministic(self):
        d = perfect(self.f,"task1")
        assert grade_task1(d,self.f) == grade_task1(d,self.f)

    @pytest.mark.parametrize("inst",["instance_001","instance_002","instance_003"])
    def test_all_instances_degenerate_differ(self, inst):
        f = load("task1",inst)
        assert abs(grade_task1(all_positive(f,"task1"),f) -
                   grade_task1(all_negative(f,"task1"),f)) > 0.05


# ══════════════════════════════════════════════════════════════════════════════
# Task 2
# ══════════════════════════════════════════════════════════════════════════════

class TestGradeTask2:
    def setup_method(self):
        self.f = load("task2")
        self.pids = paper_ids(self.f)

    def test_returns_float(self):
        assert isinstance(grade_task2({},self.f), float)
    def test_empty_low(self):
        assert 0.0 <= grade_task2({},self.f) <= 0.5
    def test_perfect_one(self):
        assert grade_task2(perfect(self.f,"task2"),self.f) == pytest.approx(1.0)
    def test_relevance_only_not_perfect(self):
        gt = self.f["ground_truth"]["labels"]
        d = {pid:{"label":gt[pid],"quality_score":1} for pid in self.pids}
        s = grade_task2(d,self.f)
        assert s < 1.0
        assert s >= 0.5
    def test_pos_neq_neg(self):
        assert abs(grade_task2(all_positive(self.f,"task2"),self.f) -
                   grade_task2(all_negative(self.f,"task2"),self.f)) > 0.05
    def test_bounds(self):
        for d in [all_positive(self.f,"task2"), all_negative(self.f,"task2"), {}]:
            assert 0.0 <= grade_task2(d,self.f) <= 1.0
    def test_missing_quality_defaults_1(self):
        gt = self.f["ground_truth"]["labels"]
        d = {pid:{"label":gt[pid]} for pid in self.pids}
        assert grade_task2(d,self.f) < 1.0
    def test_deterministic(self):
        d = perfect(self.f,"task2")
        assert grade_task2(d,self.f) == grade_task2(d,self.f)

    @pytest.mark.parametrize("inst",["instance_001","instance_002","instance_003"])
    def test_perfect_beats_neg(self, inst):
        f = load("task2",inst)
        assert grade_task2(perfect(f,"task2"),f) > grade_task2(all_negative(f,"task2"),f)


# ══════════════════════════════════════════════════════════════════════════════
# Task 3
# ══════════════════════════════════════════════════════════════════════════════

class TestGradeTask3:
    def setup_method(self):
        self.f = load("task3")
        self.pids = paper_ids(self.f)
        self.budget = 15

    def test_returns_float(self):
        assert isinstance(grade_task3({},self.f), float)
    def test_empty_zero(self):
        assert grade_task3({},self.f,steps_used=self.budget) == pytest.approx(0.0)
    def test_perfect_high(self):
        assert grade_task3(perfect(self.f,"task3"),self.f,steps_used=8) > 0.9
    def test_full_budget_no_efficiency(self):
        s = grade_task3(perfect(self.f,"task3"),self.f,steps_used=self.budget)
        assert s == pytest.approx(0.85, rel=0.05)
    def test_unused_steps_boost_score(self):
        prf = perfect(self.f,"task3")
        assert grade_task3(prf,self.f,steps_used=5) > grade_task3(prf,self.f,steps_used=15)
    def test_defer_as_exclude(self):
        gt = self.f["ground_truth"]["labels"]
        d = {pid:{"label":"DEFER" if lbl=="INCLUDE" else lbl} for pid,lbl in gt.items()}
        assert grade_task3(d,self.f,steps_used=self.budget) < 0.2
    def test_pos_neq_neg(self):
        assert abs(grade_task3(all_positive(self.f,"task3"),self.f,steps_used=self.budget) -
                   grade_task3(all_negative(self.f,"task3"),self.f,steps_used=self.budget)) > 0.05
    def test_bounds(self):
        for d in [all_positive(self.f,"task3"), all_negative(self.f,"task3"), {}]:
            assert 0.0 <= grade_task3(d,self.f,steps_used=self.budget) <= 1.0
    def test_none_steps_efficiency_zero(self):
        prf = perfect(self.f,"task3")
        s_none = grade_task3(prf,self.f,steps_used=None)
        s_full = grade_task3(prf,self.f,steps_used=self.budget)
        assert s_none == pytest.approx(s_full, rel=1e-5)
    def test_deterministic(self):
        d = perfect(self.f,"task3")
        assert grade_task3(d,self.f,steps_used=10) == grade_task3(d,self.f,steps_used=10)

    @pytest.mark.parametrize("inst",["instance_001","instance_002","instance_003"])
    def test_perfect_beats_all_pos(self, inst):
        f = load("task3",inst)
        assert grade_task3(perfect(f,"task3"),f,steps_used=10) > \
               grade_task3(all_positive(f,"task3"),f,steps_used=10)


# ══════════════════════════════════════════════════════════════════════════════
# Task 4
# ══════════════════════════════════════════════════════════════════════════════

class TestGradeTask4:
    def setup_method(self):
        self.f = load("task4")
        self.pids = paper_ids(self.f)
        self.gt = self.f["ground_truth"]

    def test_returns_float(self):
        assert isinstance(grade_task4({},self.f), float)
    def test_empty_zero(self):
        assert grade_task4({},self.f) == pytest.approx(0.0)
    def test_perfect_near_one(self):
        assert grade_task4(perfect(self.f,"task4"),self.f) > 0.95
    def test_all_neg_zero(self):
        assert grade_task4(all_negative(self.f,"task4"),self.f) == pytest.approx(0.0)
    def test_all_pos_below_perfect(self):
        assert grade_task4(perfect(self.f,"task4"),self.f) > \
               grade_task4(all_positive(self.f,"task4"),self.f)
    def test_bounds(self):
        for d in [all_positive(self.f,"task4"), all_negative(self.f,"task4"), {}]:
            assert 0.0 <= grade_task4(d,self.f) <= 1.0
    def test_rank_order_matters(self):
        prf = perfect(self.f,"task4")
        rev = {pid:dict(d) for pid,d in prf.items()}
        for pid,d in rev.items():
            if d.get("label")=="INCLUDE" and "rank" in d:
                d["rank"] = 6 - d["rank"]
        assert grade_task4(prf,self.f) > grade_task4(rev,self.f)
    def test_bad_justification_reduces_score(self):
        good = perfect(self.f,"task4")
        bad  = {pid:dict(d) for pid,d in good.items()}
        for pid,d in bad.items():
            if d.get("label")=="INCLUDE":
                d["justification"] = "zzz qqq rrr"
        assert grade_task4(good,self.f) > grade_task4(bad,self.f)
    def test_missing_rank_reduces_ndcg(self):
        prf = perfect(self.f,"task4")
        no_r = {pid:dict(d) for pid,d in prf.items()}
        for pid,d in no_r.items():
            if d.get("label")=="INCLUDE": d.pop("rank",None)
        assert grade_task4(prf,self.f) > grade_task4(no_r,self.f)
    def test_partial_correct_partial_score(self):
        gt = self.gt
        ranked = gt["ranked_order"]
        vocab = gt["vocabulary_list"]
        word = vocab[0]
        decisions = {}
        for i,pid in enumerate(ranked[:2]):
            decisions[pid]={"label":"INCLUDE","rank":i+1,
                "justification":f"{word} trained model evaluation."}
        for pid in self.pids:
            if pid not in decisions:
                decisions[pid]={"label":"EXCLUDE"}
        s = grade_task4(decisions,self.f)
        assert 0.0 < s < 1.0
    def test_deterministic(self):
        d = perfect(self.f,"task4")
        assert grade_task4(d,self.f) == grade_task4(d,self.f)

    @pytest.mark.parametrize("inst",["instance_001","instance_002","instance_003"])
    def test_all_instances_perfect_near_one(self, inst):
        f = load("task4",inst)
        assert grade_task4(perfect(f,"task4"),f) > 0.95

    @pytest.mark.parametrize("inst",["instance_001","instance_002","instance_003"])
    def test_all_instances_degenerate_differ(self, inst):
        f = load("task4",inst)
        assert abs(grade_task4(all_positive(f,"task4"),f) -
                   grade_task4(all_negative(f,"task4"),f)) > 0.05


# ══════════════════════════════════════════════════════════════════════════════
# Cross-grader invariants
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("task,fn",[
    ("task1",grade_task1),("task2",grade_task2),
    ("task3",grade_task3),("task4",grade_task4),
])
@pytest.mark.parametrize("inst",["instance_001","instance_002","instance_003"])
def test_score_in_unit_interval(task, fn, inst):
    f = load(task, inst)
    budget = f.get("budget",15)
    for d in [all_positive(f,task), all_negative(f,task), {}]:
        s = fn(d, f, steps_used=budget)
        assert 0.0 <= s <= 1.0, f"{task}/{inst}: {s}"

@pytest.mark.parametrize("task,fn",[
    ("task1",grade_task1),("task2",grade_task2),
    ("task3",grade_task3),("task4",grade_task4),
])
@pytest.mark.parametrize("inst",["instance_001","instance_002","instance_003"])
def test_no_crash_on_empty(task, fn, inst):
    f = load(task, inst)
    s = fn({}, f, steps_used=0)
    assert isinstance(s, float)

@pytest.mark.parametrize("task,fn",[
    ("task1",grade_task1),("task2",grade_task2),
    ("task3",grade_task3),("task4",grade_task4),
])
@pytest.mark.parametrize("inst",["instance_001","instance_002","instance_003"])
def test_perfect_beats_empty(task, fn, inst):
    f = load(task, inst)
    budget = f.get("budget",15)
    assert fn(perfect(f,task), f, steps_used=budget//2) > fn({}, f, steps_used=budget)

@pytest.mark.parametrize("task,fn",[
    ("task1",grade_task1),("task2",grade_task2),
    ("task3",grade_task3),("task4",grade_task4),
])
@pytest.mark.parametrize("inst",["instance_001","instance_002","instance_003"])
def test_grader_deterministic(task, fn, inst):
    f = load(task, inst)
    d = perfect(f, task)
    assert fn(d, f, steps_used=5) == fn(d, f, steps_used=5)
