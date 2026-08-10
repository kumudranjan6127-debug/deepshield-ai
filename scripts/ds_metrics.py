"""Binary detection metrics, computed one way, in one place.

The positive class is **fake**. That choice fixes what every name below
means, and it is the reason the table reads the way it does:

    FP   a real photograph the model called fake
    FPR  how often that happens — the number that decides whether
         DeepShield is usable, because a false accusation costs more
         than a missed forgery
    FN   a deepfake the model called real
    FNR  the miss rate

Everything is derived from (y_true, score) where score is P(fake).
Nothing here imports sklearn: the deployment venv does not have it, and
these functions are verified against hand-computed answers in
scripts/metrics_test.py rather than against another library.

Degenerate inputs return None rather than a number. A ROC-AUC over a set
containing only fakes is not 0.5, it is undefined, and printing 0.5 would
be a lie that looks like a measurement.
"""
import numpy as np

__all__ = ["confusion", "roc_auc", "pr_auc", "evaluate", "sweep",
           "threshold_for_fpr", "format_report"]

DEFAULT_THRESHOLD = 0.5


def _clean(y_true, score):
    y = np.asarray(y_true).astype(int).ravel()
    s = np.asarray(score, dtype=float).ravel()
    if y.shape != s.shape:
        raise ValueError(f"y_true {y.shape} and score {s.shape} differ in length")
    if y.size and not np.isin(y, (0, 1)).all():
        raise ValueError("y_true must contain only 0 (real) and 1 (fake)")
    return y, s


def _ratio(num, den):
    """A rate nobody could measure is None, not zero."""
    return float(num) / float(den) if den else None


def confusion(y_true, score, threshold=DEFAULT_THRESHOLD):
    """→ dict(tp, fp, tn, fn) with fake as the positive class."""
    y, s = _clean(y_true, score)
    pred = (s >= threshold).astype(int)
    return {
        "tp": int(((pred == 1) & (y == 1)).sum()),
        "fp": int(((pred == 1) & (y == 0)).sum()),
        "tn": int(((pred == 0) & (y == 0)).sum()),
        "fn": int(((pred == 0) & (y == 1)).sum()),
    }


def roc_auc(y_true, score):
    """Area under the ROC curve, by rank (Mann-Whitney U).

    Ties share an average rank, so a model that outputs the same score for
    everything scores 0.5 and not something better."""
    y, s = _clean(y_true, score)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if not n_pos or not n_neg:
        return None

    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks = np.empty(s.size, dtype=float)
    positions = np.arange(1, s.size + 1, dtype=float)

    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = positions[i:j + 1].mean()
        i = j + 1

    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pr_auc(y_true, score):
    """Average precision — the step-wise PR area, which does not reward a
    model for the interpolation between operating points it cannot reach."""
    y, s = _clean(y_true, score)
    n_pos = int((y == 1).sum())
    if not n_pos:
        return None

    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)

    last_of_tie = np.r_[np.diff(s) != 0, True]     # one point per distinct score
    tp, fp = tp[last_of_tie], fp[last_of_tie]

    precision = tp / (tp + fp)
    recall = tp / n_pos
    gained = recall - np.r_[0.0, recall[:-1]]
    return float((gained * precision).sum())


def evaluate(y_true, score, threshold=DEFAULT_THRESHOLD):
    """Every Phase 4B number for one set of predictions."""
    y, s = _clean(y_true, score)
    c = confusion(y, s, threshold)
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]

    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)          # sensitivity / TPR / detection rate
    specificity = _ratio(tn, tn + fp)     # TNR — how often a real photo is left alone
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall else (0.0 if precision is not None and recall is not None else None))

    return {
        "n": int(y.size), "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum()),
        "threshold": float(threshold),
        **c,
        "accuracy": _ratio(tp + tn, y.size),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "fpr": _ratio(fp, fp + tn),       # real → fake.  The one that matters.
        "fnr": _ratio(fn, fn + tp),
        "roc_auc": roc_auc(y, s),
        "pr_auc": pr_auc(y, s),
    }


def sweep(y_true, score, thresholds=None):
    """The same metrics across operating points — 0.5 is a convention, not
    a result, and the cost of moving it should be visible."""
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    return [evaluate(y_true, score, t) for t in thresholds]


def threshold_for_fpr(y_true, score, target_fpr=0.01):
    """The lowest threshold whose false-positive rate stays within target,
    with the recall it costs. → (threshold, fpr, recall) or None when there
    are no real examples to measure a false-positive rate against."""
    y, s = _clean(y_true, score)
    if not (y == 0).any() or not (y == 1).any():
        return None

    for t in np.unique(np.r_[s, 1.0 + 1e-9]):
        m = evaluate(y, s, float(t))
        if m["fpr"] is not None and m["fpr"] <= target_fpr:
            return float(t), m["fpr"], m["recall"]
    return None


# ------------------------------------------------------------------ display

def _pct(v):
    return "   n/a" if v is None else f"{v * 100:6.2f}%"


def format_report(m, title=""):
    """One metric block, aligned, with the confusion matrix that produced it."""
    head = f"  {title}" if title else ""
    return "\n".join(filter(None, [
        head,
        f"    n = {m['n']}   ({m['n_real']} real, {m['n_fake']} fake)"
        f"   threshold {m['threshold']:.2f}",
        "",
        f"      Accuracy    {_pct(m['accuracy'])}      TP {m['tp']:>6}   "
        f"FN {m['fn']:>6}   (fake)",
        f"      Precision   {_pct(m['precision'])}      FP {m['fp']:>6}   "
        f"TN {m['tn']:>6}   (real)",
        f"      Recall      {_pct(m['recall'])}",
        f"      Specificity {_pct(m['specificity'])}      ROC-AUC  "
        + ("   n/a" if m["roc_auc"] is None else f"{m['roc_auc']:.4f}"),
        f"      F1          {_pct(m['f1'])}      PR-AUC   "
        + ("   n/a" if m["pr_auc"] is None else f"{m['pr_auc']:.4f}"),
        "",
        f"      FPR         {_pct(m['fpr'])}   real called fake",
        f"      FNR         {_pct(m['fnr'])}   fake called real",
    ]))
