"""Known-answer tests for the metric core.

    python scripts/metrics_test.py

Every Phase 4 number the project reports comes out of ds_metrics.py, so a
quiet bug there would corrupt the model card rather than crash anything.
These tests do not compare against another library — they compare against
answers worked out by hand, and against a second, deliberately naive
implementation written a different way:

  roc_auc   rank formula  vs  brute-force pair counting  (O(n^2))
  pr_auc    vectorised    vs  a loop over thresholds using confusion()

Two implementations that disagree mean one of them is wrong, which is the
only kind of evidence worth having here.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ds_metrics import (confusion, evaluate, pr_auc, roc_auc, sweep,  # noqa: E402
                        threshold_for_fpr)

PASS, FAIL = [], []
TOL = 1e-12


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


def close(a, b, tol=TOL):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tol


# ------------------------------------------------- second opinions

def auc_bruteforce(y, s):
    """Every (fake, real) pair: 1 if ordered correctly, 0.5 if tied."""
    pos = [s[i] for i in range(len(y)) if y[i] == 1]
    neg = [s[i] for i in range(len(y)) if y[i] == 0]
    if not pos or not neg:
        return None
    total = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return total / (len(pos) * len(neg))


def ap_bruteforce(y, s):
    """Average precision by walking distinct thresholds through confusion()."""
    y, s = list(y), list(s)
    if not any(y):
        return None
    total, prev_recall = 0.0, 0.0
    for t in sorted(set(s), reverse=True):
        c = confusion(y, s, t)
        precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 1.0
        recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
        total += (recall - prev_recall) * precision
        prev_recall = recall
    return total


# ------------------------------------------------------------ hand-worked

def test_hand_computed():
    print("\nHand-computed answers")

    # Perfect separation
    y, s = [1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]
    m = evaluate(y, s)
    check("perfect: accuracy 100%", close(m["accuracy"], 1.0))
    check("perfect: FPR 0", close(m["fpr"], 0.0))
    check("perfect: FNR 0", close(m["fnr"], 0.0))
    check("perfect: ROC-AUC 1.0", close(m["roc_auc"], 1.0), str(m["roc_auc"]))
    check("perfect: PR-AUC 1.0", close(m["pr_auc"], 1.0), str(m["pr_auc"]))

    # Exactly backwards
    m = evaluate([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9])
    check("inverted: ROC-AUC 0.0", close(m["roc_auc"], 0.0), str(m["roc_auc"]))
    check("inverted: FPR 100%", close(m["fpr"], 1.0))

    # No information at all — every score identical
    m = evaluate([1, 1, 0, 0], [0.5, 0.5, 0.5, 0.5])
    check("all tied: ROC-AUC 0.5 (not better)", close(m["roc_auc"], 0.5), str(m["roc_auc"]))

    # Worked out by hand:
    #   pairs (fake,real): (.9,.6)=1  (.9,.1)=1  (.4,.6)=0  (.4,.1)=1  → 3/4
    #   AP: order .9(F) .6(R) .4(F) .1(R)
    #       precision 1, .5, 2/3, .5 ; recall .5, .5, 1, 1
    #       0.5*1 + 0*.5 + 0.5*(2/3) + 0*.5 = 0.8333...
    y, s = [1, 1, 0, 0], [0.9, 0.4, 0.6, 0.1]
    check("mixed: ROC-AUC 0.75", close(roc_auc(y, s), 0.75), str(roc_auc(y, s)))
    check("mixed: PR-AUC 5/6", close(pr_auc(y, s), 0.5 + (1 / 3)), str(pr_auc(y, s)))

    # A confusion matrix counted by eye
    #   scores  .9 .8 .3   (fake)      .7 .2 .1   (real)      threshold .5
    #   fake ≥ .5 → 2 TP, 1 FN ;  real ≥ .5 → 1 FP, 2 TN
    y = [1, 1, 1, 0, 0, 0]
    s = [0.9, 0.8, 0.3, 0.7, 0.2, 0.1]
    m = evaluate(y, s, 0.5)
    check("counted by eye: TP=2 FN=1 FP=1 TN=2",
          (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 2),
          f"{m['tp']} {m['fn']} {m['fp']} {m['tn']}")
    check("counted by eye: precision 2/3", close(m["precision"], 2 / 3))
    check("counted by eye: recall 2/3", close(m["recall"], 2 / 3))
    check("counted by eye: specificity 2/3", close(m["specificity"], 2 / 3))
    check("counted by eye: F1 2/3", close(m["f1"], 2 / 3), str(m["f1"]))
    check("counted by eye: FPR 1/3", close(m["fpr"], 1 / 3))
    check("counted by eye: FNR 1/3", close(m["fnr"], 1 / 3))


def test_definitions_hold():
    print("\nDefinitions that must hold on any input")
    rng = np.random.default_rng(20260810)
    for trial in range(200):
        n = int(rng.integers(4, 60))
        y = rng.integers(0, 2, n)
        if y.sum() in (0, n):                 # keep both classes present
            y[0], y[-1] = 1, 0
        s = np.round(rng.random(n), 2)        # rounding forces plenty of ties
        m = evaluate(y, s, float(rng.choice([0.3, 0.5, 0.7])))

        if not close(m["fpr"], 1 - m["specificity"], 1e-12):
            check(f"trial {trial}: FPR == 1 - specificity", False)
            return
        if not close(m["fnr"], 1 - m["recall"], 1e-12):
            check(f"trial {trial}: FNR == 1 - recall", False)
            return
        if m["tp"] + m["fp"] + m["tn"] + m["fn"] != n:
            check(f"trial {trial}: confusion matrix sums to n", False)
            return
    check("FPR == 1 - specificity  (200 random sets)", True)
    check("FNR == 1 - recall  (200 random sets)", True)
    check("confusion matrix always sums to n  (200 random sets)", True)


def test_second_implementation():
    print("\nAgainst a second, independent implementation")
    rng = np.random.default_rng(4242)
    worst_auc = worst_ap = 0.0
    for _ in range(120):
        n = int(rng.integers(4, 80))
        y = rng.integers(0, 2, n)
        if y.sum() in (0, n):
            y[0], y[-1] = 1, 0
        s = np.round(rng.random(n), 1)        # heavy ties on purpose
        worst_auc = max(worst_auc, abs(roc_auc(y, s) - auc_bruteforce(list(y), list(s))))
        worst_ap = max(worst_ap, abs(pr_auc(y, s) - ap_bruteforce(list(y), list(s))))

    check("ROC-AUC matches brute-force pair counting", worst_auc < 1e-12,
          f"worst disagreement {worst_auc:.2e}")
    check("PR-AUC matches a threshold walk", worst_ap < 1e-12,
          f"worst disagreement {worst_ap:.2e}")


def test_rank_invariance():
    print("\nProperties")
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, 40)
    y[0], y[-1] = 1, 0
    s = rng.random(40)
    base = roc_auc(y, s)
    squashed = roc_auc(y, 1 / (1 + np.exp(-6 * (s - 0.5))))    # monotone
    check("ROC-AUC survives a monotone rescale of the scores",
          close(base, squashed, 1e-12), f"{base:.6f} vs {squashed:.6f}")

    # Raising the threshold can only ever reduce false positives
    rows = sweep(y, s)
    fps = [r["fp"] for r in rows]
    check("false positives never rise as the threshold rises",
          all(a >= b for a, b in zip(fps, fps[1:])), str(fps))
    recalls = [r["recall"] for r in rows]
    check("recall never rises as the threshold rises",
          all(a >= b - 1e-12 for a, b in zip(recalls, recalls[1:])))


def test_degenerate():
    print("\nUndefined is reported as undefined, not as a number")
    m = evaluate([1, 1, 1], [0.9, 0.8, 0.7])
    check("only fakes: ROC-AUC is None", m["roc_auc"] is None, str(m["roc_auc"]))
    check("only fakes: specificity is None", m["specificity"] is None)
    check("only fakes: FPR is None", m["fpr"] is None)
    check("only fakes: recall still measurable", close(m["recall"], 1.0))

    m = evaluate([0, 0, 0], [0.1, 0.2, 0.3])
    check("only reals: PR-AUC is None", m["pr_auc"] is None)
    check("only reals: recall is None", m["recall"] is None)
    check("only reals: FPR still measurable", close(m["fpr"], 0.0))

    m = evaluate([], [])
    check("empty set does not crash", m["n"] == 0 and m["accuracy"] is None)

    check("threshold_for_fpr needs both classes",
          threshold_for_fpr([1, 1], [0.9, 0.8]) is None)

    # Mismatched lengths and stray labels are refused, not silently truncated
    for bad, why in (((["1", 2], [0.5]), "length mismatch"),
                     (([0, 2], [0.1, 0.9]), "label outside {0,1}")):
        try:
            evaluate(*bad)
            check(f"rejects {why}", False, "no error raised")
        except ValueError:
            check(f"rejects {why}", True)


def test_threshold_for_fpr():
    print("\nOperating point selection")
    #  reals at .10 .20 .30 .40   fakes at .35 .60 .70 .80
    #  threshold .41 → no real flagged (FPR 0), fake at .35 missed → recall 3/4
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    s = [0.10, 0.20, 0.30, 0.40, 0.35, 0.60, 0.70, 0.80]
    got = threshold_for_fpr(y, s, target_fpr=0.0)
    check("finds a zero-FPR operating point", got is not None)
    if got:
        t, fpr, recall = got
        check("that point really has FPR 0", close(fpr, 0.0), f"threshold {t}")
        check("and costs the fake that scored below every real",
              close(recall, 0.75), f"recall {recall}")
    got = threshold_for_fpr(y, s, target_fpr=0.25)
    check("a looser FPR budget buys back recall",
          got is not None and got[2] >= 0.75, str(got))


def main():
    print("DeepShield metric tests")
    test_hand_computed()
    test_definitions_hold()
    test_second_implementation()
    test_rank_invariance()
    test_degenerate()
    test_threshold_for_fpr()

    total = len(PASS) + len(FAIL)
    print("\n" + "=" * 52)
    print(f"passed {len(PASS)} / {total}")
    if FAIL:
        print("\nFAILED:")
        for f in FAIL:
            print("  - " + f)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
