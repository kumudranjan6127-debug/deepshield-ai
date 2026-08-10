"""Leakage tests for the V4 training split.

The split that matters lives inside a Kaggle notebook, which cannot run
here — no GPU, no datasets. So this lifts the actual split code out of the
`.ipynb` and executes it against a synthetic DFDC whose leakage structure is
known by construction.

That structure is the whole point. In DFDC one real video is the source of
several fakes, so three files can show the same face:

    abc.mp4  REAL   original: -        -> group abc
    def.mp4  FAKE   original: abc.mp4  -> group abc
    ghi.mp4  FAKE   original: abc.mp4  -> group abc

Split those by filename and the model is tested on a face it trained on.
"""
import ast
import io
import json
import os
import random
import re

import pytest

pytestmark = pytest.mark.metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK = os.path.join(ROOT, "training", "DeepShield_V4_Universal.ipynb")

START = "from collections import defaultdict"
END = "rng.shuffle(dfdc_fake); rng.shuffle(dfdc_real)"


@pytest.fixture(scope="session")
def cells():
    if not os.path.exists(NOTEBOOK):
        pytest.skip("the V4 notebook is not present")
    nb = json.load(io.open(NOTEBOOK, encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


@pytest.fixture(scope="session")
def split_code(cells):
    """The real split logic, straight out of the notebook."""
    for source in cells:
        if START in source and END in source:
            return source[source.index(START):source.index(END) + len(END)]
    pytest.fail(f"could not find the split block in {NOTEBOOK}")


def synthetic_dfdc(n_real=200, seed=1):
    """(rows, truth) — every fake tagged with the real video it came from."""
    rnd = random.Random(seed)
    rows, truth = [], {}
    for i in range(n_real):
        stem = f"real{i:04d}"
        rows.append((f"/faces/{stem}.jpg", 1, stem))
        truth[f"/faces/{stem}.jpg"] = stem
        for j in range(rnd.randint(0, 4)):              # 0-4 fakes per source
            fake = f"fake{i:04d}_{j}"
            rows.append((f"/faces/{fake}.jpg", 0, stem))
            truth[f"/faces/{fake}.jpg"] = stem
    rnd.shuffle(rows)
    return rows, truth


def run_split(split_code, rows, holdout_n=100, seed=42):
    scope = {"DFDC_ROWS": rows, "rng": random.Random(seed),
             "DFDC_HOLDOUT_N": holdout_n}
    exec(compile(split_code, "<notebook split>", "exec"), scope)
    return scope


# ------------------------------------------------------------- the notebook

def test_every_notebook_cell_is_valid_python(cells):
    broken = []
    for i, source in enumerate(cells):
        clean = re.sub(r"^\s*[!%].*$", "", source, flags=re.M)   # strip !pip / %magic
        try:
            ast.parse(clean)
        except SyntaxError as exc:
            broken.append(f"cell {i}: line {exc.lineno}: {exc.msg}")
    assert not broken, "; ".join(broken)


def test_the_split_groups_come_from_the_metadata(split_code):
    assert "for p, y, g in DFDC_ROWS" in split_code, \
        "the split is grouping by something other than the identity column"
    for name in ("dfdc_by_group", "holdout_groups", "train_groups"):
        assert name in split_code


def test_the_notebook_checks_itself_at_runtime(cells):
    joined = "\n".join(cells)
    assert "Leakage check" in joined and "LEAK:" in joined
    assert "are also in training" in joined
    assert "a DFDC identity appears in both training and the holdout" in joined
    assert "UNSEEN_FAMILY" in joined and "unseen_pool" in joined
    assert "predictions_all.csv" in joined and "p_fake" in joined


# ------------------------------------------------------------- no leakage

def test_no_face_appears_on_both_sides(split_code):
    rows, truth = synthetic_dfdc()
    scope = run_split(split_code, rows)

    train_groups = set(scope["train_groups"])
    holdout_groups = set(scope["holdout_groups"])
    assert not (train_groups & holdout_groups)

    train_paths = set(scope["dfdc_fake"]) | set(scope["dfdc_real"])
    holdout_paths = {p for p, _ in scope["dfdc_holdout"]}
    assert not (train_paths & holdout_paths)

    # The real assertion: no FACE overlaps, whatever the filenames say
    shared = {truth[p] for p in train_paths} & {truth[p] for p in holdout_paths}
    assert not shared, f"{len(shared)} identities leaked"


def test_the_fix_is_not_a_no_op(split_code):
    """Measure how badly the previous file-level split leaked on identical
    data, so this cannot quietly decay into doing nothing."""
    rows, truth = synthetic_dfdc()

    rnd = random.Random(42)
    fakes = [p for p, y, _ in rows if y == 0]
    reals = [p for p, y, _ in rows if y == 1]
    rnd.shuffle(fakes)
    rnd.shuffle(reals)
    old_holdout = set(fakes[:50]) | set(reals[:50])
    old_train = set(fakes[50:]) | set(reals[50:])
    leaked = {truth[p] for p in old_train} & {truth[p] for p in old_holdout}
    assert leaked, "the synthetic set no longer reproduces the original bug"

    scope = run_split(split_code, rows)
    now = ({truth[p] for p in set(scope["dfdc_fake"]) | set(scope["dfdc_real"])} &
           {truth[p] for p, _ in scope["dfdc_holdout"]})
    assert not now, f"{len(leaked)} leaked before, {len(now)} now"


def test_the_holdout_is_still_worth_measuring_on(split_code):
    rows, _ = synthetic_dfdc()
    scope = run_split(split_code, rows, holdout_n=100)
    holdout = scope["dfdc_holdout"]

    n_fake = sum(1 for _, y in holdout if y == 0)
    assert n_fake > 0 and len(holdout) - n_fake > 0, "the holdout lost a class"
    assert len(holdout) >= 100
    assert len(scope["train_groups"]) > len(scope["holdout_groups"])
    assert all(g in scope["holdout_groups"] for _, _, g in scope["dfdc_holdout_meta"])


def test_the_split_is_deterministic(split_code):
    rows, _ = synthetic_dfdc()
    a = run_split(split_code, rows, seed=42)
    b = run_split(split_code, rows, seed=42)
    c = run_split(split_code, rows, seed=7)
    assert a["holdout_groups"] == b["holdout_groups"]
    assert a["holdout_groups"] != c["holdout_groups"]


def test_an_absent_or_tiny_dfdc_does_not_break_it(split_code):
    empty = run_split(split_code, [], holdout_n=100)
    assert not empty["dfdc_holdout"] and not empty["dfdc_fake"]
    assert not empty["train_groups"]

    tiny = run_split(split_code, [("/faces/r0.jpg", 1, "r0"),
                                  ("/faces/f0.jpg", 0, "r0")], holdout_n=100)
    used = ({p for p, _ in tiny["dfdc_holdout"]} |
            set(tiny["dfdc_fake"]) | set(tiny["dfdc_real"]))
    assert len(used) == 2
