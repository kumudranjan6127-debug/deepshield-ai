"""The evaluation arithmetic, against answers worked out by hand.

Every number this project publishes comes out of `scripts/ds_metrics.py`, so
a quiet bug there corrupts the model card rather than crashing anything.

These do not compare against scikit-learn. Agreeing with another library
tests agreement, not correctness — so the checks are hand-computed answers
plus a second implementation written a deliberately different way:

    roc_auc   rank formula   vs   brute-force pair counting  (O(n^2))
    pr_auc    vectorised     vs   a loop through confusion()
"""
import os
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from ds_metrics import (band_accuracy, brier, confusion, ece,  # noqa: E402
                        evaluate, mce, pr_auc, reliability, roc_auc, sweep,
                        threshold_for_fpr)


# ------------------------------------------------------- second opinions

def auc_bruteforce(y, s):
    """Every (fake, real) pair: 1 if ordered right, 0.5 if tied."""
    pos = [s[i] for i in range(len(y)) if y[i] == 1]
    neg = [s[i] for i in range(len(y)) if y[i] == 0]
    if not pos or not neg:
        return None
    return sum(1.0 if p > n else 0.5 if p == n else 0.0
               for p in pos for n in neg) / (len(pos) * len(neg))


def ap_bruteforce(y, s):
    """Average precision by walking distinct thresholds through confusion()."""
    y, s = list(y), list(s)
    if not any(y):
        return None
    total = previous_recall = 0.0
    for threshold in sorted(set(s), reverse=True):
        c = confusion(y, s, threshold)
        precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 1.0
        recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
        total += (recall - previous_recall) * precision
        previous_recall = recall
    return total


# ------------------------------------------------------------ hand-worked

def test_perfect_separation():
    m = evaluate([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1])
    assert m["accuracy"] == 1.0
    assert m["fpr"] == 0.0 and m["fnr"] == 0.0
    assert m["roc_auc"] == 1.0 and m["pr_auc"] == 1.0


def test_exactly_backwards():
    m = evaluate([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9])
    assert m["roc_auc"] == 0.0
    assert m["fpr"] == 1.0


def test_no_information_at_all():
    """Identical scores everywhere must score 0.5, not something better."""
    assert roc_auc([1, 1, 0, 0], [0.5] * 4) == 0.5


def test_a_mixed_case_by_hand():
    #  pairs (fake,real): (.9,.6)=1  (.9,.1)=1  (.4,.6)=0  (.4,.1)=1  -> 3/4
    #  AP: order .9(F) .6(R) .4(F) .1(R)
    #      precision 1, .5, 2/3, .5 ; recall .5, .5, 1, 1  -> 0.5 + 1/3
    y, s = [1, 1, 0, 0], [0.9, 0.4, 0.6, 0.1]
    assert roc_auc(y, s) == pytest.approx(0.75)
    assert pr_auc(y, s) == pytest.approx(0.5 + 1 / 3)


def test_a_confusion_matrix_counted_by_eye():
    #  fake .9 .8 .3   real .7 .2 .1   threshold .5
    #  fake >= .5 -> 2 TP, 1 FN ;  real >= .5 -> 1 FP, 2 TN
    m = evaluate([1, 1, 1, 0, 0, 0], [0.9, 0.8, 0.3, 0.7, 0.2, 0.1], 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 2)
    for key in ("precision", "recall", "specificity", "f1"):
        assert m[key] == pytest.approx(2 / 3)
    assert m["fpr"] == pytest.approx(1 / 3)
    assert m["fnr"] == pytest.approx(1 / 3)


# ------------------------------------------------------------- properties

def test_definitions_hold_on_any_input():
    rng = np.random.default_rng(20260810)
    for _ in range(200):
        n = int(rng.integers(4, 60))
        y = rng.integers(0, 2, n)
        if y.sum() in (0, n):
            y[0], y[-1] = 1, 0
        s = np.round(rng.random(n), 2)          # rounding forces plenty of ties
        m = evaluate(y, s, float(rng.choice([0.3, 0.5, 0.7])))

        assert m["fpr"] == pytest.approx(1 - m["specificity"])
        assert m["fnr"] == pytest.approx(1 - m["recall"])
        assert m["tp"] + m["fp"] + m["tn"] + m["fn"] == n


def test_against_a_second_implementation():
    rng = np.random.default_rng(4242)
    worst_auc = worst_ap = 0.0
    for _ in range(120):
        n = int(rng.integers(4, 80))
        y = rng.integers(0, 2, n)
        if y.sum() in (0, n):
            y[0], y[-1] = 1, 0
        s = np.round(rng.random(n), 1)          # heavy ties on purpose
        worst_auc = max(worst_auc, abs(roc_auc(y, s) - auc_bruteforce(list(y), list(s))))
        worst_ap = max(worst_ap, abs(pr_auc(y, s) - ap_bruteforce(list(y), list(s))))

    assert worst_auc < 1e-12, f"ROC-AUC disagrees by {worst_auc:.2e}"
    assert worst_ap < 1e-12, f"PR-AUC disagrees by {worst_ap:.2e}"


def test_auc_survives_a_monotone_rescale():
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, 40)
    y[0], y[-1] = 1, 0
    s = rng.random(40)
    assert roc_auc(y, s) == pytest.approx(
        roc_auc(y, 1 / (1 + np.exp(-6 * (s - 0.5)))))


def test_raising_the_threshold_only_ever_helps_specificity():
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, 40)
    y[0], y[-1] = 1, 0
    rows = sweep(y, rng.random(40))
    fps = [r["fp"] for r in rows]
    recalls = [r["recall"] for r in rows]
    assert all(a >= b for a, b in zip(fps, fps[1:]))
    assert all(a >= b - 1e-12 for a, b in zip(recalls, recalls[1:]))


# ------------------------------------------------------------- degenerate

def test_undefined_is_reported_as_undefined():
    """A ROC-AUC over one class is not 0.5. Printing a number there would
    be a lie that looks like a measurement."""
    only_fakes = evaluate([1, 1, 1], [0.9, 0.8, 0.7])
    assert only_fakes["roc_auc"] is None
    assert only_fakes["specificity"] is None and only_fakes["fpr"] is None
    assert only_fakes["recall"] == 1.0

    only_reals = evaluate([0, 0, 0], [0.1, 0.2, 0.3])
    assert only_reals["pr_auc"] is None and only_reals["recall"] is None
    assert only_reals["fpr"] == 0.0

    empty = evaluate([], [])
    assert empty["n"] == 0 and empty["accuracy"] is None
    assert threshold_for_fpr([1, 1], [0.9, 0.8]) is None


@pytest.mark.parametrize("args", [(["1", 2], [0.5]), ([0, 2], [0.1, 0.9])])
def test_bad_input_is_refused_not_truncated(args):
    with pytest.raises(ValueError):
        evaluate(*args)


def test_an_operating_point_can_be_chosen_for_a_fpr_budget():
    #  reals at .10 .20 .30 .40   fakes at .35 .60 .70 .80
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    s = [0.10, 0.20, 0.30, 0.40, 0.35, 0.60, 0.70, 0.80]
    threshold, fpr, recall = threshold_for_fpr(y, s, target_fpr=0.0)
    assert fpr == 0.0
    assert recall == pytest.approx(0.75), "the fake below every real should be missed"
    assert threshold_for_fpr(y, s, target_fpr=0.25)[2] >= 0.75


# ------------------------------------------------------------ calibration

def test_calibration_hand_cases():
    assert brier([1, 1, 0, 0], [1, 1, 0, 0]) == 0.0
    assert ece([1, 1, 0, 0], [1, 1, 0, 0]) == 0.0
    assert brier([1, 1, 0, 0], [0, 0, 1, 1]) == 1.0
    assert ece([1, 1, 0, 0], [0, 0, 1, 1]) == 1.0
    assert brier([1, 0], [1.0, 1.0]) == 0.5
    assert ece([1, 0], [1.0, 1.0]) == 0.5


def test_calibrated_and_useless_are_different_things():
    """Answering 0.5 to a balanced set is perfectly calibrated and carries
    no information. Calibration and discrimination are not the same."""
    y, s = [1, 1, 0, 0], [0.5, 0.5, 0.5, 0.5]
    assert brier(y, s) == pytest.approx(0.25)
    assert ece(y, s) == pytest.approx(0.0)
    assert roc_auc(y, s) == 0.5


def test_confidence_mode_measures_the_number_the_ui_shows():
    #  four predictions reported at 0.9 confidence, three right
    #  claimed 0.90, delivered 0.75 -> gap 0.15
    y, s = [1, 1, 1, 0], [0.9, 0.9, 0.9, 0.9]
    assert ece(y, s, mode="confidence") == pytest.approx(0.15)
    rows = reliability(y, s, mode="confidence")
    assert len(rows) == 1
    assert rows[0]["observed"] == pytest.approx(0.75)


def test_calibration_properties():
    rng = np.random.default_rng(11)
    for _ in range(50):
        n = int(rng.integers(4, 80))
        y, s = rng.integers(0, 2, n), rng.random(n)
        assert 0.0 <= brier(y, s) <= 1.0
        assert ece(y, s) <= mce(y, s) + 1e-12

    rows = reliability(rng.integers(0, 2, 60), rng.random(60), bins=10)
    assert sum(r["n"] for r in rows) == 60
    assert reliability([], []) == []
    assert brier([], []) is None


# ---------------------------------------------------------- certainty bands

def test_band_accuracy_by_hand():
    from config import CFG
    #  y=1 p=.95 -> conf 95 fake right   y=0 p=.95 -> conf 95 fake WRONG
    #  y=1 p=.75 -> conf 75 fake right   y=1 p=.55 -> conf 55 fake right
    rows = {r["key"]: r for r in
            band_accuracy([1, 0, 1, 1], [0.95, 0.95, 0.75, 0.55],
                          CFG.CERTAINTY_BANDS)}
    assert rows["very_strong"]["n"] == 2
    assert rows["very_strong"]["accuracy"] == pytest.approx(0.5)
    assert rows["strong"]["n"] == 1 and rows["strong"]["accuracy"] == 1.0
    assert rows["uncertain"]["n"] == 1
    assert rows["low_evidence"]["n"] == 0
    assert rows["low_evidence"]["accuracy"] is None, "an empty band invented a rate"
    assert sum(r["n"] for r in rows.values()) == 4


def test_the_lowest_band_is_unreachable_by_construction():
    """`confidence` is max(p, 1-p) for two classes, so it never drops below
    50 and the 0-30 band cannot occur — whatever the model does. This is
    KNOWN_ISSUES #3, measured rather than argued."""
    from config import CFG
    rng = np.random.default_rng(3)
    rows = {r["key"]: r for r in
            band_accuracy(rng.integers(0, 2, 4000), rng.random(4000),
                          CFG.CERTAINTY_BANDS)}
    assert rows["low_evidence"]["n"] == 0
    assert rows["uncertain"]["n"] > 0
