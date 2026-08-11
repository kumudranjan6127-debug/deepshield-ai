"""Fetch a real-face set to measure the false-positive rate against.

    python scripts/fetch_real_faces.py                 # 500 people, 1 photo each
    python scripts/fetch_real_faces.py --count 1000
    python scripts/fetch_real_faces.py --per-person 2

**Why not FFHQ.** FFHQ is the entire real class this model trained on.
Measuring the false-positive rate against it would measure how well the
model memorised its own training distribution — the one number that is
already known and the one that means least.

So: LFW (Labeled Faces in the Wild), 13,233 photographs of 5,749 people
collected from news and the web. Different source, different cameras,
different decades of JPEG. Out-of-domain by construction, which is the
whole point.

**What LFW is not.** These are press and web photographs, not pictures off
a modern phone. They are 250x250, already loosely cropped, and carry the
compression history of 2000s web publishing. A phone-camera set would be a
better proxy for the app's actual traffic and is still missing. This
measures one honest out-of-domain condition, not every condition.

One photo per person by default. LFW has 530 pictures of George W. Bush;
counting those as 530 independent tests would inflate the sample size by an
order of magnitude and tell you about one face.
"""
import argparse
import csv
import json
import os
import random
import socket
import sys
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "eval_data", "real", "lfw")
GROUPS_CSV = os.path.join(ROOT, "eval_data", "groups.csv")

DATASET = "bitmind/lfw"
ROWS_API = "https://datasets-server.huggingface.co/rows"
PAGE = 100
TOTAL_ROWS = 13233


def fetch_json(url, tries=3):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))


def person_of(filename):
    """`George_W_Bush_0042.jpg` → `George_W_Bush`. The identity group."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    parts = stem.rsplit("_", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else stem


def download(url, dest, tries=3):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                blob = r.read()
            if len(blob) < 512:
                return False
            with open(dest, "wb") as f:
                f.write(blob)
            return True
        except Exception:
            if attempt == tries - 1:
                return False
            time.sleep(1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--count", type=int, default=500, help="how many photographs")
    ap.add_argument("--per-person", type=int, default=1,
                    help="photos per identity; >1 shrinks the effective sample")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    socket.setdefaulttimeout(30)
    os.makedirs(OUT_DIR, exist_ok=True)

    rng = random.Random(args.seed)
    # Sample across the whole set rather than taking the first N: LFW is
    # ordered by name, so the first 500 rows are people whose names begin
    # with A.
    offsets = list(range(0, TOTAL_ROWS - PAGE, PAGE))
    rng.shuffle(offsets)

    print(f"DeepShield — fetching {args.count} real photographs from {DATASET}")
    seen, chosen = {}, []
    for offset in offsets:
        if len(chosen) >= args.count:
            break
        url = (f"{ROWS_API}?dataset={urllib.parse.quote(DATASET)}"
               f"&config=default&split=train&offset={offset}&length={PAGE}")
        try:
            page = fetch_json(url)
        except Exception as exc:
            print(f"  page at {offset} failed: {type(exc).__name__}")
            continue

        for entry in page.get("rows", []):
            row = entry.get("row", {})
            name = row.get("filename") or ""
            src = (row.get("image") or {}).get("src")
            if not name or not src:
                continue
            person = person_of(name)
            if seen.get(person, 0) >= args.per_person:
                continue
            seen[person] = seen.get(person, 0) + 1
            chosen.append((person, name, src))
            if len(chosen) >= args.count:
                break
        print(f"\r  listed {len(chosen)}/{args.count} "
              f"({len(seen)} people)", end="", flush=True)
    print()

    rows, failed = [], 0
    for i, (person, name, src) in enumerate(chosen, 1):
        dest = os.path.join(OUT_DIR, f"{person}__{i:04d}.jpg")
        if os.path.exists(dest) or download(src, dest):
            rows.append({"path": os.path.relpath(dest, ROOT).replace(os.sep, "/"),
                         "group": person})
        else:
            failed += 1
        if i % 25 == 0 or i == len(chosen):
            print(f"\r  downloaded {len(rows)}/{len(chosen)}", end="", flush=True)
    print()

    if not rows:
        sys.exit("nothing downloaded — check the network and try again")

    # Merge into groups.csv rather than replacing it: the fake side already
    # has its own entries and they must not be lost.
    existing = []
    if os.path.exists(GROUPS_CSV):
        with open(GROUPS_CSV, newline="", encoding="utf-8") as f:
            existing = [r for r in csv.DictReader(f)
                        if not r.get("path", "").replace("\\\\", "/").startswith(
                            "eval_data/real/lfw/")]
    with open(GROUPS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "group"])
        w.writeheader()
        w.writerows(existing + rows)

    people = len({r["group"] for r in rows})
    print(f"\n  {len(rows)} photographs of {people} distinct people -> "
          f"{os.path.relpath(OUT_DIR, ROOT)}")
    if failed:
        print(f"  {failed} could not be downloaded")
    print(f"  groups.csv now has {len(existing) + len(rows)} rows")
    print("\n  Now measure it:")
    print("    python scripts/evaluate.py --data eval_data --target-fpr 0.01")
    return 0


if __name__ == "__main__":
    sys.exit(main())
