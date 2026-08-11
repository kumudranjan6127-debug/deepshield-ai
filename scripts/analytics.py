"""Read what the deployment has collected.

    DATABASE_URL=... python scripts/analytics.py
    DATABASE_URL=... python scripts/analytics.py --days 7
    DATABASE_URL=... python scripts/analytics.py --export disagreements.csv

Deliberately a script and not an endpoint. An `/api/analytics` route would
be readable by anyone who guessed the path, and gating it would mean
building the backend authentication this project does not have. A script
run by whoever holds the connection string needs neither.

**The disagreement list is the reason this exists.** Every row where someone
said the verdict was wrong is a candidate mislabel on a real photograph —
which is the labelled data the false-positive rate, the calibration curve
and the video weights have all been waiting for.
"""
import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--export", metavar="FILE",
                    help="write the disagreements to CSV for review")
    args = ap.parse_args()

    import store
    from config import CFG

    print(f"DeepShield analytics — backend: {store.backend_name()}")
    if not store.enabled():
        print("\n  No database configured.")
        print("  Set DATABASE_URL and install psycopg:  pip install \"psycopg[binary]\"")
        if os.path.exists(CFG.FEEDBACK_PATH):
            with open(CFG.FEEDBACK_PATH, encoding="utf-8") as f:
                lines = [l for l in f if l.strip()]
            print(f"\n  Local file has {len(lines)} feedback row(s): "
                  f"{os.path.relpath(CFG.FEEDBACK_PATH, ROOT)}")
            print("  On an ephemeral host that file is erased every time the "
                  "service sleeps, which is what the database is for.")
        return 0

    data = store.summary(args.days)
    if not data.get("available"):
        print(f"\n  Could not read: {data.get('error', 'unknown')}")
        return 1

    print(f"\nLast {data['window_days']} days")
    print("=" * 52)
    print(f"  analyses          {data['analyses']:,}")
    print(f"  feedback          {data['feedback']:,}")
    print(f"  disagreements     {data['disagreements']:,}", end="")
    if "disagreement_rate" in data:
        print(f"   ({data['disagreement_rate'] * 100:.1f}% of feedback)")
    else:
        print()

    if data.get("by_prediction"):
        print("\n  verdicts")
        for key, count in sorted(data["by_prediction"].items(), key=lambda kv: -kv[1]):
            print(f"    {key:<12} {count:>6,}")

    if data.get("by_file_type"):
        print("\n  media")
        for key, count in sorted(data["by_file_type"].items(), key=lambda kv: -kv[1]):
            print(f"    {key:<12} {count:>6,}")

    latency = data.get("latency_ms") or {}
    if latency.get("max"):
        print(f"\n  latency           mean {latency['mean']:,} ms   "
              f"max {latency['max']:,} ms")

    rows = data.get("recent_disagreements") or []
    print("\n" + "=" * 52)
    print("DISAGREEMENTS — the rows worth looking at")
    print("=" * 52)
    if not rows:
        print("  none yet.")
        print("\n  Nothing here until people use the deployment and tell it when")
        print("  it is wrong. Those answers are the labelled data this project")
        print("  has never had — see KNOWN_ISSUES #1, #2 and #4.")
    else:
        print(f"  {'scan':<22}{'said':<12}{'conf':>6}  {'kind':<8}when")
        print("  " + "-" * 60)
        for r in rows:
            when = (r.get("at") or "")[:16].replace("T", " ")
            print(f"  {str(r.get('scan_id'))[:20]:<22}{str(r.get('prediction')):<12}"
                  f"{r.get('confidence') or 0:>5}%  {str(r.get('file_type')):<8}{when}")

        print("\n  A disagreement is a claim, not a fact — someone may simply be")
        print("  wrong. Read them before treating any as a label.")

    if args.export and rows:
        with open(args.export, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  exported -> {args.export}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
