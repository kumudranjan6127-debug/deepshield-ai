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

__all__ = [
    "abstention_summary",
    "apply_temperature",
    "band_accuracy",
    "bootstrap_confidence_intervals",
    "brier",
    "confusion",
    "decide_with_abstention",
    "ece",
    "evaluate",
    "fit_temperature",
    "format_bands",
    "format_calibration",
    "format_report",
    "logits_from_probabilities",
    "mce",
    "pr_auc",
    "reliability",
    "roc_auc",
    "select_abstention_thresholds",
    "sweep",
    "threshold_for_fpr",
]

DEFAULT_THRESHOLD = 0.5


def _clean(y_true, score):
    y = np.asarray(y_true).astype(int).ravel()
    s = np.asarray(score, dtype=float).ravel()
    if y.shape != s.shape:
        raise ValueError(f"y_true {y.shape} and score {s.shape} differ in length")
    if y.size and not np.isin(y, (0, 1)).all():
        raise ValueError("y_true must contain only 0 (real) and 1 (fake)")
    return y, s


def _clean_probability(y_true, score):
    y, s = _clean(y_true, score)
    if not np.isfinite(s).all() or ((s < 0) | (s > 1)).any():
        raise ValueError("scores must be finite values in [0, 1]")
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


# ------------------------------------------------------- V5 calibration and
#                                                       abstention policy

def logits_from_probabilities(score, epsilon=1e-6):
    """Safely recover binary logits from an uncalibrated P(fake) score.

    Temperature scaling acts on logits. An exported ONNX model may make
    those available directly; when it does not, the log-odds transform is
    mathematically equivalent for a two-class softmax. Clipping avoids an
    infinite logit from a rounded 0 or 1 without changing the ordering.
    """
    s = np.asarray(score, dtype=float).ravel()
    if not np.isfinite(s).all() or ((s < 0) | (s > 1)).any():
        raise ValueError("probabilities must be finite values in [0, 1]")
    if not 0 < epsilon < 0.5:
        raise ValueError("epsilon must be between 0 and 0.5")
    p = np.clip(s, epsilon, 1.0 - epsilon)
    return np.log(p) - np.log1p(-p)


def _sigmoid(values):
    values = np.asarray(values, dtype=float)
    # Stable for both saturated ONNX logits and p=0/1 transformed above.
    positive = values >= 0
    out = np.empty_like(values)
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out


def _nll(y, logits):
    return float(np.mean(np.logaddexp(0.0, logits) - y * logits))


def fit_temperature(y_true, *, logits=None, probabilities=None,
                    minimum=0.05, maximum=10.0):
    """Fit one temperature on the explicitly held-out calibration set.

    The implementation uses scalar golden-section minimisation rather than
    adding scipy/torch to the small production environment. It returns a
    serialisable record containing enough provenance to audit how a future
    score was transformed; callers must record it with the calibration split,
    never with validation or sealed-test rows.
    """
    if (logits is None) == (probabilities is None):
        raise ValueError("provide exactly one of logits or probabilities")
    if not 0 < minimum < maximum:
        raise ValueError("temperature bounds must satisfy 0 < minimum < maximum")
    if probabilities is not None:
        y, p = _clean_probability(y_true, probabilities)
        raw_logits = logits_from_probabilities(p)
        input_kind = "probabilities_logit_transform"
        before = p
    else:
        y = np.asarray(y_true).astype(int).ravel()
        raw_logits = np.asarray(logits, dtype=float).ravel()
        if y.shape != raw_logits.shape:
            raise ValueError("y_true and logits differ in length")
        if y.size and not np.isin(y, (0, 1)).all():
            raise ValueError("y_true must contain only 0 (real) and 1 (fake)")
        if not np.isfinite(raw_logits).all():
            raise ValueError("logits must be finite")
        input_kind = "stored_logits"
        before = _sigmoid(raw_logits)
    if not y.size or not (y == 0).any() or not (y == 1).any():
        raise ValueError("temperature fitting requires both real and fake calibration samples")

    lo, hi = np.log(minimum), np.log(maximum)
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    c = hi - (hi - lo) / phi
    d = lo + (hi - lo) / phi
    for _ in range(80):
        if _nll(y, raw_logits / np.exp(c)) <= _nll(y, raw_logits / np.exp(d)):
            hi = d
        else:
            lo = c
        c = hi - (hi - lo) / phi
        d = lo + (hi - lo) / phi
    temperature = float(np.exp((lo + hi) / 2.0))
    after = _sigmoid(raw_logits / temperature)
    return {
        "method": "temperature_scaling",
        "temperature": temperature,
        "input": input_kind,
        "n": int(y.size),
        "n_real": int((y == 0).sum()),
        "n_fake": int((y == 1).sum()),
        "nll_before": _nll(y, raw_logits),
        "nll_after": _nll(y, raw_logits / temperature),
        "brier_before": brier(y, before),
        "brier_after": brier(y, after),
        "ece_before": ece(y, before),
        "ece_after": ece(y, after),
    }


def apply_temperature(*, logits=None, probabilities=None, temperature):
    """Apply a previously fitted temperature without re-fitting it."""
    if (logits is None) == (probabilities is None):
        raise ValueError("provide exactly one of logits or probabilities")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a positive finite number")
    raw = (np.asarray(logits, dtype=float).ravel() if logits is not None
           else logits_from_probabilities(probabilities))
    if not np.isfinite(raw).all():
        raise ValueError("logits must be finite")
    return _sigmoid(raw / float(temperature))


def decide_with_abstention(score, real_threshold, fake_threshold):
    """Classify only when the score clears a real/fake evidence boundary."""
    scores = np.asarray(score, dtype=float).ravel()
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("scores must be finite values in [0, 1]")
    if not 0 <= real_threshold < fake_threshold <= 1:
        raise ValueError("thresholds must satisfy 0 <= real < fake <= 1")
    return np.where(scores <= real_threshold, "real",
                    np.where(scores >= fake_threshold, "fake", "inconclusive"))


def abstention_summary(y_true, score, real_threshold, fake_threshold):
    """Decision coverage and conclusive confusion matrix for a three-way policy."""
    y, s = _clean_probability(y_true, score)
    verdict = decide_with_abstention(s, real_threshold, fake_threshold)
    conclusive = verdict != "inconclusive"
    predicted = (verdict[conclusive] == "fake").astype(int)
    actual = y[conclusive]
    return {
        "n": int(y.size),
        "conclusive": int(conclusive.sum()),
        "inconclusive": int((~conclusive).sum()),
        "coverage": float(conclusive.mean()) if y.size else None,
        "inconclusive_rate": float((~conclusive).mean()) if y.size else None,
        "conclusive_metrics": evaluate(actual, predicted.astype(float), 0.5),
    }


def select_abstention_thresholds(y_true, score, target_fpr=0.01,
                                 target_fnr=0.01, minimum_band=0.10):
    """Choose real/fake cut points using validation data only.

    ``minimum_band`` reserves an inconclusive interval around 0.5. Lowering
    the real boundary and raising the fake boundary can only improve the two
    controlled error budgets, so the interval is never narrowed merely to
    make a validation number look better.
    """
    y, s = _clean_probability(y_true, score)
    if not y.size or not (y == 0).any() or not (y == 1).any():
        raise ValueError("threshold selection requires both real and fake validation samples")
    if not 0 <= target_fpr <= 1 or not 0 <= target_fnr <= 1:
        raise ValueError("target FPR and FNR must be in [0, 1]")
    if not 0 <= minimum_band < 1:
        raise ValueError("minimum_band must be in [0, 1)")

    fake_point = threshold_for_fpr(y, s, target_fpr)
    fake_threshold = float(fake_point[0]) if fake_point else 1.0
    fake_threshold = min(1.0, max(0.5, fake_threshold))
    actual_fpr = float((s[y == 0] >= fake_threshold).mean())
    if actual_fpr > target_fpr:
        raise ValueError(
            "no fake threshold in [0, 1] satisfies the validation FPR budget")

    # A standard threshold calls scores *below* it real. Take the greatest
    # one that preserves the validation FNR budget, then make room for the
    # abstention band conservatively.
    real_candidates = sorted(set(np.r_[0.0, s[s <= 0.5]]))
    fake_scores = s[y == 1]
    # The three-way policy calls scores *equal* to this boundary real. Using
    # evaluate(..., threshold=t) here would count equality as fake and could
    # silently exceed the requested FNR budget at the chosen boundary.
    feasible = [
        t for t in real_candidates
        if float((fake_scores <= t).mean()) <= target_fnr
    ]
    if not feasible:
        raise ValueError(
            "no real threshold in [0, 1] satisfies the validation FNR budget")
    real_threshold = float(max(feasible))
    half_band = minimum_band / 2.0
    real_threshold = min(real_threshold, 0.5 - half_band)
    fake_threshold = max(fake_threshold, 0.5 + half_band)
    policy = abstention_summary(y, s, real_threshold, fake_threshold)
    policy.update({
        "source_split": "validation",
        "real_threshold": real_threshold,
        "fake_threshold": fake_threshold,
        "target_fpr": float(target_fpr),
        "target_fnr": float(target_fnr),
        "minimum_band": float(minimum_band),
    })
    return policy


def bootstrap_confidence_intervals(y_true, score, threshold=DEFAULT_THRESHOLD,
                                   resamples=200, confidence=0.95, seed=42,
                                   groups=None):
    """Non-parametric CIs, resampling groups when their ids are available."""
    y, s = _clean_probability(y_true, score)
    if resamples < 0:
        raise ValueError("resamples must be non-negative")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    metric_names = ("accuracy", "precision", "recall", "specificity", "f1",
                    "fpr", "fnr", "roc_auc", "pr_auc", "brier", "ece")
    estimate = evaluate(y, s, threshold)
    estimate.update({"brier": brier(y, s), "ece": ece(y, s)})
    result = {name: {"estimate": estimate.get(name), "lower": None, "upper": None,
                     "valid_resamples": 0} for name in metric_names}
    if not resamples or not y.size:
        if groups is None:
            independent_units = int(y.size)
        else:
            group_values = np.asarray(groups, dtype=object).ravel()
            if group_values.shape != y.shape:
                raise ValueError("groups must have one value per sample")
            independent_units = len({
                (str(group).strip()
                 if group is not None and str(group).strip()
                 else f"__row_{index}")
                for index, group in enumerate(group_values)
            })
        return {"resamples": int(resamples), "confidence": float(confidence),
                "unit": "groups" if groups is not None else "samples",
                "independent_units": independent_units, "metrics": result}

    rng = np.random.default_rng(seed)
    if groups is None:
        units = [np.array([i], dtype=int) for i in range(y.size)]
        unit_name = "samples"
    else:
        group_values = np.asarray(groups, dtype=object).ravel()
        if group_values.shape != y.shape:
            raise ValueError("groups must have one value per sample")
        grouped = {}
        for index, group in enumerate(group_values):
            # Missing groups do not become one correlated pseudo-person.
            key = (str(group).strip()
                   if group is not None and str(group).strip()
                   else f"__row_{index}")
            grouped.setdefault(key, []).append(index)
        units = [np.asarray(indexes, dtype=int) for indexes in grouped.values()]
        unit_name = "groups"
    values = {name: [] for name in metric_names}
    for _ in range(resamples):
        sampled_units = rng.integers(0, len(units), len(units))
        indexes = np.concatenate([units[int(index)] for index in sampled_units])
        row = evaluate(y[indexes], s[indexes], threshold)
        row.update({"brier": brier(y[indexes], s[indexes]), "ece": ece(y[indexes], s[indexes])})
        for name in metric_names:
            if row[name] is not None:
                values[name].append(float(row[name]))
    low = (1.0 - confidence) / 2.0 * 100.0
    high = (1.0 + confidence) / 2.0 * 100.0
    for name, series in values.items():
        if series:
            result[name].update({"lower": float(np.percentile(series, low)),
                                 "upper": float(np.percentile(series, high)),
                                 "valid_resamples": len(series)})
    return {"resamples": int(resamples), "confidence": float(confidence),
            "unit": unit_name, "independent_units": len(units), "metrics": result}


# -------------------------------------------------------------- calibration
#
# Discrimination and calibration are different questions, and a model can
# be excellent at one while useless at the other.
#
#   ROC-AUC asks: does it rank fakes above reals?
#   Calibration asks: when it says 0.9, is it right 90% of the time?
#
# A model trained with cross-entropy and picked by validation accuracy is
# usually badly calibrated and badly overconfident. That is exactly why the
# UI should not say "94% probability" — until the numbers below are
# measured, the percentage is a ranking dressed up as a frequency.

def brier(y_true, score):
    """Mean squared error of the probability. 0 is perfect; 0.25 is what
    you get by answering 0.5 to everything, so anything above 0.25 is
    worse than admitting you do not know."""
    y, s = _clean_probability(y_true, score)
    return float(((s - y) ** 2).mean()) if y.size else None


def reliability(y_true, score, bins=10, mode="positive"):
    """Rows of a reliability diagram — the data behind ECE.

    mode='positive'    bin by P(fake); compare it to how often those were
                       actually fake. The classic diagram.
    mode='confidence'  bin by max(p, 1-p) — the number the UI shows —
                       and compare it to how often the verdict was right.
                       This is the one that validates the certainty bands.

    → [{lo, hi, n, mean_score, observed, gap}]; empty bins are dropped.
    """
    y, s = _clean_probability(y_true, score)
    if not y.size:
        return []

    if mode == "confidence":
        value = np.maximum(s, 1 - s)
        correct = ((s >= 0.5).astype(int) == y).astype(float)
        lo_edge = 0.5                      # a two-class confidence cannot go lower
    elif mode == "positive":
        value, correct, lo_edge = s, y.astype(float), 0.0
    else:
        raise ValueError("mode must be 'positive' or 'confidence'")

    edges = np.linspace(lo_edge, 1.0, bins + 1)
    rows = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        # last bin closes on the right so a score of exactly 1.0 is counted
        inside = (value >= lo) & (value < hi) if i < bins - 1 else \
                 (value >= lo) & (value <= hi)
        n = int(inside.sum())
        if not n:
            continue
        mean_score = float(value[inside].mean())
        observed = float(correct[inside].mean())
        rows.append({"lo": float(lo), "hi": float(hi), "n": n,
                     "mean_score": mean_score, "observed": observed,
                     "gap": observed - mean_score})
    return rows


def ece(y_true, score, bins=10, mode="positive"):
    """Expected calibration error — the average gap between what was
    claimed and what happened, weighted by how many predictions landed in
    each bin."""
    rows = reliability(y_true, score, bins, mode)
    total = sum(r["n"] for r in rows)
    if not total:
        return None
    return float(sum(r["n"] * abs(r["gap"]) for r in rows) / total)


def mce(y_true, score, bins=10, mode="positive"):
    """The worst bin. ECE can look healthy while one region is far out."""
    rows = reliability(y_true, score, bins, mode)
    return max((abs(r["gap"]) for r in rows), default=None)


def band_accuracy(y_true, score, bands):
    """Observed accuracy inside each certainty band.

    `bands` is the table the product actually uses — [(lower, key, label)],
    highest first. This is how a band label stops being a guess: if the
    band called "Strong evidence" is right 61% of the time, the label is
    wrong and the cut point moves.

    → [{key, label, from, to, n, accuracy}] in the order given.
    """
    y, s = _clean(y_true, score)
    confidence = np.round(np.maximum(s, 1 - s) * 100).astype(int)
    correct = (s >= 0.5).astype(int) == y

    out = []
    for i, (lower, key, label) in enumerate(bands):
        upper = bands[i - 1][0] if i else 101
        inside = (confidence >= lower) & (confidence < upper)
        n = int(inside.sum())
        out.append({"key": key, "label": label, "from": int(lower),
                    "to": int(min(upper, 100)), "n": n,
                    "accuracy": float(correct[inside].mean()) if n else None})
    return out


# ------------------------------------------------------------------ display

def _pct(v):
    return "   n/a" if v is None else f"{v * 100:6.2f}%"


def format_report(m, title=""):
    """One metric block, aligned, with the confusion matrix that produced it."""
    head = f"  {title}" if title else ""
    return "\n".join(filter(None, [
        head,
        (f"    n = {m['n']}   ({m['n_real']} real, {m['n_fake']} fake)"
         f"   threshold {m['threshold']:.2f}"),
        "",
        (f"      Accuracy    {_pct(m['accuracy'])}      TP {m['tp']:>6}   "
         f"FN {m['fn']:>6}   (fake)"),
        (f"      Precision   {_pct(m['precision'])}      FP {m['fp']:>6}   "
         f"TN {m['tn']:>6}   (real)"),
        f"      Recall      {_pct(m['recall'])}",
        f"      Specificity {_pct(m['specificity'])}      ROC-AUC  "
        + ("   n/a" if m["roc_auc"] is None else f"{m['roc_auc']:.4f}"),
        f"      F1          {_pct(m['f1'])}      PR-AUC   "
        + ("   n/a" if m["pr_auc"] is None else f"{m['pr_auc']:.4f}"),
        "",
        f"      FPR         {_pct(m['fpr'])}   real called fake",
        f"      FNR         {_pct(m['fnr'])}   fake called real",
    ]))


def format_calibration(y_true, score, bins=10, mode="positive"):
    """A reliability diagram that survives a terminal.

    A perfectly calibrated model has an empty gap column. Bars point the
    way the model was wrong: `<` claimed more than it delivered."""
    rows = reliability(y_true, score, bins, mode)
    if not rows:
        return "    (no predictions to plot)"

    claimed = "claimed P(fake)" if mode == "positive" else "reported confidence"
    happened = "actually fake" if mode == "positive" else "verdict correct"

    lines = [f"    {claimed:>19}        n   {happened:>16}       gap",
             "    " + "-" * 72]
    for r in rows:
        bar = ("<" if r["gap"] < 0 else ">") * round(abs(r["gap"]) * 40)
        lines.append(f"    {r['lo']:>8.2f} - {r['hi']:.2f} {r['n']:>8}   "
                     f"{r['observed'] * 100:>15.1f}%   {r['gap']:+.3f}  {bar}")

    e, m = ece(y_true, score, bins, mode), mce(y_true, score, bins, mode)
    lines += ["",
              f"    ECE {e:.4f}    MCE {m:.4f}    Brier {brier(y_true, score):.4f}",
              "    '<' = over-confident: the model claimed more than happened."]
    return "\n".join(lines)


def format_bands(rows):
    """Certainty bands against what actually happened inside them.

    An empty band is worth as much attention as a wrong one: it means the
    label can never be produced, and a vocabulary with unreachable words in
    it is describing something other than this model."""
    lines = ["    band                     range        n    accuracy",
             "    " + "-" * 56]
    for r in rows:
        acc = "      n/a" if r["accuracy"] is None else f"{r['accuracy'] * 100:8.2f}%"
        note = "" if r["n"] else "   <- never occurs"
        lines.append(f"    {r['label']:22s} {r['from']:>3}-{r['to']:<4} {r['n']:>8}  "
                     f"{acc}{note}")
    return "\n".join(lines)
