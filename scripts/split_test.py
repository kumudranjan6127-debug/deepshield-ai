"""Leakage tests for the V4 training split.

    python scripts/split_test.py

The split that matters lives inside a Kaggle notebook, which cannot be run
here — no GPU, no datasets. So this test does the next best thing: it
lifts the actual split code out of the .ipynb and executes it against a
synthetic DFDC whose leakage structure is known by construction.

That structure is the whole point. In DFDC one real video is the source of
several fakes, so three files can show the same face:

    abc.mp4  REAL   original: -        -> group abc
    def.mp4  FAKE   original: abc.mp4  -> group abc
    ghi.mp4  FAKE   original: abc.mp4  -> group abc

Split those by filename and the model is tested on a face it trained on.
The test asserts the notebook keeps them together, and — so the fix is not
mistaken for a no-op — also measures how badly the previous file-level
split leaked on identical data.
"""
import io
import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK = os.path.join(ROOT, "training", "DeepShield_V4_Universal.ipynb")

START = "from collections import defaultdict"
END = "rng.shuffle(dfdc_fake); rng.shuffle(dfdc_real)"

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  - {detail}" if detail else ""))
    return ok


def notebook_cells():
    nb = json.load(io.open(NOTEBOOK, encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def split_code():
    """The real split logic, straight out of the notebook."""
    for src in notebook_cells():
        if START in src and END in src:
            return src[src.index(START):src.index(END) + len(END)]
    sys.exit(f"could not find the split block in {NOTEBOOK}\n"
             f"  looked for a cell containing both:\n    {START}\n    {END}")


def synthetic_dfdc(n_real=200, seed=1):
    """(rows, truth) — every fake tagged with the real video it came from."""
    rnd = random.Random(seed)
    rows, truth = [], {}
    for i in range(n_real):
        stem = f"real{i:04d}"
        rows.append((f"/faces/{stem}.jpg", 1, stem))
        truth[f"/faces/{stem}.jpg"] = stem
        for j in range(rnd.randint(0, 4)):                 # 0-4 fakes per source
            fake = f"fake{i:04d}_{j}"
            rows.append((f"/faces/{fake}.jpg", 0, stem))   # same identity group
            truth[f"/faces/{fake}.jpg"] = stem
    rnd.shuffle(rows)
    return rows, truth


def run_split(rows, holdout_n=100, seed=42):
    scope = {"DFDC_ROWS": rows, "rng": random.Random(seed),
             "DFDC_HOLDOUT_N": holdout_n}
    exec(compile(split_code(), "<notebook split>", "exec"), scope)
    return scope


# ------------------------------------------------------------------- tests

def test_extraction():
    print("\nThe notebook still contains the code this test checks")
    code = split_code()
    check("split block located in the notebook", bool(code),
          f"{len(code.splitlines())} lines")
    for needed in ("dfdc_by_group", "holdout_groups", "train_groups"):
        check(f"block defines {needed}", needed in code)
    check("groups come from DFDC_ROWS, not filenames",
          "for p, y, g in DFDC_ROWS" in code)


def test_no_identity_leak():
    print("\nNo face appears on both sides")
    rows, truth = synthetic_dfdc()
    s = run_split(rows)

    train_groups = set(s["train_groups"])
    holdout_groups = set(s["holdout_groups"])
    check("no identity group is in both sets", not (train_groups & holdout_groups),
          f"{len(train_groups)} train / {len(holdout_groups)} holdout groups")

    train_paths = set(s["dfdc_fake"]) | set(s["dfdc_real"])
    holdout_paths = {p for p, _ in s["dfdc_holdout"]}
    check("no file is in both sets", not (train_paths & holdout_paths))

    # The real assertion: the FACES must not overlap, whatever the filenames
    train_faces = {truth[p] for p in train_paths}
    holdout_faces = {truth[p] for p in holdout_paths}
    shared = train_faces & holdout_faces
    check("no FACE is in both sets", not shared,
          f"{len(shared)} leaked identities" if shared else
          f"{len(train_faces)} trained, {len(holdout_faces)} held out")


def test_fix_is_not_a_no_op():
    print("\nThe old file-level split really did leak on the same data")
    rows, truth = synthetic_dfdc()

    # Exactly what the notebook used to do: shuffle files, cut at N/2
    rnd = random.Random(42)
    fakes = [p for p, y, _ in rows if y == 0]
    reals = [p for p, y, _ in rows if y == 1]
    rnd.shuffle(fakes); rnd.shuffle(reals)
    half = 100 // 2
    old_holdout = set(fakes[:half]) | set(reals[:half])
    old_train = set(fakes[half:]) | set(reals[half:])

    leaked = {truth[p] for p in old_train} & {truth[p] for p in old_holdout}
    check("file-level split leaked identities", bool(leaked),
          f"{len(leaked)} of {len(old_holdout)} held-out files shared a face "
          f"with training")

    s = run_split(rows)
    now = ({truth[p] for p in set(s["dfdc_fake"]) | set(s["dfdc_real"])} &
           {truth[p] for p, _ in s["dfdc_holdout"]})
    check("group-level split leaks none", not now,
          f"{len(leaked)} leaked before, {len(now)} now")


def test_holdout_is_usable():
    print("\nThe holdout is still worth measuring on")
    rows, _ = synthetic_dfdc()
    s = run_split(rows, holdout_n=100)
    holdout = s["dfdc_holdout"]
    n_fake = sum(1 for _, y in holdout if y == 0)
    n_real = len(holdout) - n_fake

    check("holdout has both classes", n_fake > 0 and n_real > 0,
          f"{n_fake} fake, {n_real} real")
    check("holdout reaches the requested size", len(holdout) >= 100,
          f"{len(holdout)} crops from {len(s['holdout_groups'])} identities")
    check("training keeps the large majority of identities",
          len(s["train_groups"]) > len(s["holdout_groups"]),
          f"{len(s['train_groups'])} vs {len(s['holdout_groups'])}")
    check("every held-out crop belongs to a held-out group",
          all(g in s["holdout_groups"] for _, _, g in s["dfdc_holdout_meta"]))


def test_deterministic():
    print("\nThe same seed gives the same split")
    rows, _ = synthetic_dfdc()
    a, b = run_split(rows, seed=42), run_split(rows, seed=42)
    check("identical seeds agree", a["holdout_groups"] == b["holdout_groups"],
          f"{len(a['holdout_groups'])} groups")
    c = run_split(rows, seed=7)
    check("a different seed gives a different split",
          a["holdout_groups"] != c["holdout_groups"])


def test_degenerate_input():
    print("\nAn absent or tiny DFDC does not break the notebook")
    s = run_split([], holdout_n=100)
    check("no DFDC at all: empty split, no crash",
          not s["dfdc_holdout"] and not s["dfdc_fake"] and not s["train_groups"])

    tiny = [("/faces/r0.jpg", 1, "r0"), ("/faces/f0.jpg", 0, "r0")]
    s = run_split(tiny, holdout_n=100)
    used = {p for p, _ in s["dfdc_holdout"]} | set(s["dfdc_fake"]) | set(s["dfdc_real"])
    check("fewer images than requested: still consistent", len(used) == 2,
          f"{len(s['dfdc_holdout'])} held out, "
          f"{len(s['dfdc_fake']) + len(s['dfdc_real'])} trainable")


def test_leakage_cell_exists():
    print("\nThe notebook checks itself at runtime")
    cells = notebook_cells()
    joined = "\n".join(cells)
    check("a leakage-check cell is present",
          "Leakage check" in joined and "LEAK:" in joined)
    check("it asserts on files", "are also in training" in joined)
    check("it asserts on identities",
          "a DFDC identity appears in both training and the holdout" in joined)
    check("a whole generator family is withheld",
          "UNSEEN_FAMILY" in joined and "unseen_pool" in joined)
    check("evaluation writes predictions for the repo to score",
          "predictions_all.csv" in joined and "p_fake" in joined)


def test_cells_parse():
    print("\nEvery cell is valid Python")
    import ast
    bad = []
    for i, src in enumerate(notebook_cells()):
        clean = re.sub(r"^\s*[!%].*$", "", src, flags=re.M)   # strip !pip / %magic
        try:
            ast.parse(clean)
        except SyntaxError as exc:
            bad.append(f"code cell {i}: line {exc.lineno}: {exc.msg}")
    check(f"all {len(notebook_cells())} code cells parse", not bad, "; ".join(bad))


def main():
    print(f"V4 split tests  ({os.path.relpath(NOTEBOOK, ROOT)})")
    test_extraction()
    test_cells_parse()
    test_no_identity_leak()
    test_fix_is_not_a_no_op()
    test_holdout_is_usable()
    test_deterministic()
    test_degenerate_input()
    test_leakage_cell_exists()

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
