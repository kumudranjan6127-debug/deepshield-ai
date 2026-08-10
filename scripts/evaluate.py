"""Score a labelled image set with the live engine and report every metric.

    python scripts/evaluate.py                      # scores eval_data/
    python scripts/evaluate.py --data DIR --seen ffhq,sg1,tpdn,diffusion
    python scripts/evaluate.py --from-csv preds.csv # recompute, no model needed
    python scripts/evaluate.py --conditions SRC --out eval_data/real/processed

Expected layout — the folder under each class names the source, and the
source is what the per-source table reports on:

    eval_data/
      real/ffhq/*.jpg          real/phone/*.jpg      real/screenshot/*.png
      fake/stylegan/*.jpg      fake/dfdc/*.jpg       fake/diffusion/*.jpg

Two things this deliberately does:

**It scores through `inference.score_image`**, the same preprocessing a
real upload gets. A benchmark that runs its own resize pipeline measures a
model the users never meet.

**It computes nothing itself.** Every number comes from `ds_metrics`, so
the figures printed here and the figures a Kaggle notebook produces are
the same arithmetic — the notebook writes a predictions CSV, and
`--from-csv` turns it into the report.

Images only. Video is scored frame-by-frame by the app; a per-frame
benchmark needs a different labelling scheme than this one.
"""
import argparse
import csv
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ds_metrics as M  # noqa: E402

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
FIELDS = ["path", "source", "label", "y_true", "p_fake", "group"]


# --------------------------------------------------------------- collecting

def group_of(path):
    """Which images must never be split across train and test.

    Defaults to the filename stem with any `__cond-*` suffix removed, so
    the four processed variants of one photograph count as one source
    image rather than four independent samples. A `groups.csv`
    (path,group) next to the data overrides this — DFDC needs it, because
    every fake made from the same original video shares that original."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.split("__cond-")[0]


def load_group_overrides(data_dir):
    path = os.path.join(data_dir, "groups.csv")
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {os.path.normcase(r["path"]): r["group"]
                for r in csv.DictReader(f) if r.get("path")}


def collect(data_dir):
    """→ [(path, source, label, y_true)] walking real/<source> and fake/<source>."""
    items = []
    for label, y_true in (("real", 0), ("fake", 1)):
        root = os.path.join(data_dir, label)
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            rel = os.path.relpath(dirpath, root)
            source = "(root)" if rel == "." else rel.replace(os.sep, "/").split("/")[0]
            for name in sorted(filenames):
                if name.lower().endswith(IMAGE_EXT):
                    items.append((os.path.join(dirpath, name), source, label, y_true))
    return items


# ----------------------------------------------------------------- scoring

def score_all(items, overrides, out_csv, limit=None):
    import inference

    if not inference.engine_available():
        sys.exit("no model available - nothing to evaluate")
    if limit:
        items = items[:limit]

    rows, failures = [], []
    started = time.time()
    for i, (path, source, label, y_true) in enumerate(items, 1):
        try:
            p_fake = inference.score_image(path)
        except Exception as exc:
            failures.append((path, f"{type(exc).__name__}: {exc}"))
            continue
        rows.append({
            "path": os.path.relpath(path, ROOT).replace(os.sep, "/"),
            "source": source, "label": label, "y_true": y_true,
            "p_fake": f"{p_fake:.6f}",
            "group": overrides.get(os.path.normcase(path)) or group_of(path),
        })
        if i % 25 == 0 or i == len(items):
            rate = i / max(time.time() - started, 1e-6)
            print(f"\r  scored {i}/{len(items)}  ({rate:.1f} img/s)", end="", flush=True)
    print()

    if out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"  predictions -> {os.path.relpath(out_csv, ROOT)}")

    if failures:
        print(f"\n  {len(failures)} image(s) could not be scored:")
        for path, why in failures[:10]:
            print(f"    {os.path.basename(path)}: {why}")
    return rows


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    missing = [c for c in ("y_true", "p_fake") if rows and c not in rows[0]]
    if missing:
        sys.exit(f"{path} is missing column(s): {', '.join(missing)}")
    for r in rows:
        r.setdefault("source", "(all)")
        r.setdefault("label", "fake" if int(r["y_true"]) else "real")
    return rows


# ------------------------------------------------------------------ report

def split(rows):
    return [int(r["y_true"]) for r in rows], [float(r["p_fake"]) for r in rows]


def per_source_table(rows, threshold):
    """The dataset matrix, one line per source.

    A real source can only be judged on false positives and a fake source
    only on detection, so the last column changes meaning by class and
    says which it is."""
    sources = {}
    for r in rows:
        sources.setdefault((r["label"], r["source"]), []).append(r)

    lines = ["    class  source                 n    mean P(fake)   verdict",
             "    " + "-" * 66]
    for (label, source), group in sorted(sources.items()):
        y, s = split(group)
        m = M.evaluate(y, s, threshold)
        mean_p = sum(s) / len(s)
        if label == "real":
            rate = m["fpr"]
            verdict = f"{rate * 100:6.2f}% called fake" if rate is not None else "n/a"
        else:
            rate = m["recall"]
            verdict = f"{rate * 100:6.2f}% detected" if rate is not None else "n/a"
        lines.append(f"    {label:5s}  {source:20s} {len(group):5d}"
                     f"      {mean_p:.3f}      {verdict}")
    return "\n".join(lines)


def sweep_table(rows):
    y, s = split(rows)
    lines = ["    threshold   accuracy   recall    FPR       F1",
             "    " + "-" * 48]
    for m in M.sweep(y, s):
        def pct(v):
            return "   n/a  " if v is None else f"{v * 100:6.2f}%"
        lines.append(f"      {m['threshold']:.2f}      {pct(m['accuracy'])}  "
                     f"{pct(m['recall'])}  {pct(m['fpr'])}  {pct(m['f1'])}")
    return "\n".join(lines)


def group_note(rows):
    """Repeated groups mean correlated samples: a set of 500 crops from 50
    videos is closer to 50 independent tests than 500, and saying so keeps
    the headline number honest."""
    groups = {r.get("group") for r in rows if r.get("group")}
    if not groups or len(groups) == len(rows):
        return ""
    return (f"\n  {len(rows)} images come from {len(groups)} independent "
            f"groups (person / video / source).\n  Treat the sample size as "
            f"{len(groups)}, not {len(rows)} — samples inside a group are "
            "correlated.")


def report(rows, threshold, seen=None, target_fpr=0.01):
    if not rows:
        sys.exit("no predictions to report on")

    y, s = split(rows)
    print("\n" + "=" * 70)
    print("OVERALL")
    print("=" * 70)
    print(M.format_report(M.evaluate(y, s, threshold)))
    print(group_note(rows))

    print("\n" + "=" * 70)
    print("BY SOURCE  - the dataset matrix")
    print("=" * 70)
    print(per_source_table(rows, threshold))

    print("\n" + "=" * 70)
    print("THRESHOLD SWEEP  - 0.5 is a convention, not a result")
    print("=" * 70)
    print(sweep_table(rows))

    calibration(rows)

    point = M.threshold_for_fpr(y, s, target_fpr)
    if point:
        t, fpr, recall = point
        print(f"\n  For a false-positive rate at or below {target_fpr * 100:g}%:")
        print(f"    threshold {t:.3f}  ->  FPR {fpr * 100:.2f}%  "
              f"detection {recall * 100:.2f}%")
    else:
        print(f"\n  No threshold reaches an FPR of {target_fpr * 100:g}% "
              "(or one class is missing).")

    if seen:
        cross_dataset(rows, threshold, seen)


def calibration(rows):
    """Whether the percentage the UI prints means anything.

    Two questions, deliberately both asked. The first is about the raw
    probability; the second is about the number a user is actually shown,
    and it is the one that decides whether a certainty band deserves its
    name."""
    y, s = split(rows)

    print("\n" + "=" * 70)
    print("CALIBRATION  - is the number a frequency or just a ranking?")
    print("=" * 70)
    print(M.format_calibration(y, s, mode="positive"))
    print()
    print(M.format_calibration(y, s, mode="confidence"))

    try:
        sys.path.insert(0, os.path.join(ROOT, "backend"))
        from config import CFG
    except Exception:
        return

    print("\n" + "=" * 70)
    print("CERTAINTY BANDS  - the labels against what actually happened")
    print("=" * 70)
    print(M.format_bands(M.band_accuracy(y, s, CFG.CERTAINTY_BANDS)))
    print("\n  A band whose observed accuracy is far from its name is a band")
    print("  whose cut point is wrong. These are the numbers that should")
    print("  replace CERTAINTY_BANDS in backend/config.py.")


def cross_dataset(rows, threshold, seen):
    """In-domain against held-out generators.

    The gap between these two blocks is the honest headline. A detector
    that scores 99% on the generators it trained against and 60% on one it
    has never seen is a 60% detector as far as the real world is
    concerned."""
    seen = {x.strip().lower() for x in seen if x.strip()}
    in_domain = [r for r in rows if r["source"].lower() in seen]
    unseen = [r for r in rows if r["source"].lower() not in seen]

    print("\n" + "=" * 70)
    print("CROSS-DATASET  - trained-on vs never-seen")
    print("=" * 70)
    if not in_domain or not unseen:
        print("    Needs both: sources named in --seen, and sources outside it.")
        print(f"    in-domain {len(in_domain)} images, unseen {len(unseen)}.")
        return

    for title, subset in (("IN-DOMAIN  " + ", ".join(sorted(seen)), in_domain),
                          ("UNSEEN  " + ", ".join(sorted(
                              {r['source'] for r in unseen})), unseen)):
        y, s = split(subset)
        print()
        print(M.format_report(M.evaluate(y, s, threshold), title))

    a = M.evaluate(*split(in_domain), threshold)
    b = M.evaluate(*split(unseen), threshold)
    if a["accuracy"] is not None and b["accuracy"] is not None:
        print(f"\n  Generalisation gap: {(a['accuracy'] - b['accuracy']) * 100:+.2f} "
              "points of accuracy")
    if a["recall"] is not None and b["recall"] is not None:
        print(f"  Detection gap:      {(a['recall'] - b['recall']) * 100:+.2f} "
              "points of recall")


# ------------------------------------------------------- condition variants

CONDITIONS = {
    # name          long side   save as   quality
    "orig":        (None,       "JPEG",   95),
    "phone":       (1440,       "JPEG",   92),
    "screenshot":  (1080,       "PNG",    None),
    "social":      (720,        "JPEG",   60),
    "reencode":    (None,       "JPEG",   40),
}


def make_conditions(src_dir, out_dir):
    """Write processed variants of every image in src_dir.

    These approximate what happens to a photograph on its way through a
    phone, a screenshot, a messaging app and repeated forwarding. They are
    stand-ins with plausible parameters, not measurements of any specific
    platform's pipeline — the point is to show how the verdict moves as
    evidence is destroyed, which is where a detector's real false-positive
    rate lives."""
    from PIL import Image

    files = [os.path.join(src_dir, f) for f in sorted(os.listdir(src_dir))
             if f.lower().endswith(IMAGE_EXT)]
    if not files:
        sys.exit(f"no images in {src_dir}")

    made = 0
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        for name, (side, fmt, quality) in CONDITIONS.items():
            target = os.path.join(out_dir, name,
                                  f"{stem}__cond-{name}.{'png' if fmt == 'PNG' else 'jpg'}")
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with Image.open(path) as im:
                im = im.convert("RGB")
                if side and max(im.size) > side:
                    scale = side / max(im.size)
                    im = im.resize((max(1, round(im.width * scale)),
                                    max(1, round(im.height * scale))),
                                   Image.LANCZOS)
                if name == "reencode":                    # forwarded twice
                    import io
                    buf = io.BytesIO()
                    im.save(buf, "JPEG", quality=55)
                    buf.seek(0)
                    im = Image.open(buf).convert("RGB")
                im.save(target, fmt, **({"quality": quality} if quality else {}))
            made += 1

    print(f"  {made} variants of {len(files)} image(s) -> {out_dir}")
    print(f"  conditions: {', '.join(CONDITIONS)}")
    print("  Variants of one photo share a group, so they count as one sample.")


# -------------------------------------------------------------------- entry

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=os.path.join(ROOT, "eval_data"),
                    help="folder holding real/<source> and fake/<source>")
    ap.add_argument("--from-csv", help="skip scoring, report on an existing CSV")
    ap.add_argument("--out", default=os.path.join(ROOT, "eval_data", "predictions.csv"),
                    help="where to write predictions")
    ap.add_argument("--threshold", type=float, default=M.DEFAULT_THRESHOLD)
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--seen", help="comma-separated sources the model trained on")
    ap.add_argument("--limit", type=int, help="score only the first N images")
    ap.add_argument("--conditions", metavar="SRC_DIR",
                    help="generate processed variants instead of evaluating")
    args = ap.parse_args()

    if args.conditions:
        out = args.out
        if out.endswith(".csv"):                  # --out defaults to the CSV path
            out = os.path.join(ROOT, "eval_data", "conditions")
        return make_conditions(args.conditions, out)

    print("DeepShield evaluation")

    if args.from_csv:
        rows = read_csv(args.from_csv)
        print(f"  {len(rows)} predictions from {args.from_csv}")
    else:
        if not os.path.isdir(args.data):
            sys.exit(f"no evaluation data at {args.data}\n"
                     f"See eval_data/README.md for the expected layout.")
        items = collect(args.data)
        if not items:
            sys.exit(f"{args.data} has no images under real/ or fake/")
        import inference
        info = inference.engine_info()
        print(f"  model  {info.get('model_name')} {info.get('version')} "
              f"({info.get('runtime')})")
        print(f"  found  {len(items)} images")
        rows = score_all(items, load_group_overrides(args.data), args.out, args.limit)

    report(rows, args.threshold,
           seen=args.seen.split(",") if args.seen else None,
           target_fpr=args.target_fpr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
