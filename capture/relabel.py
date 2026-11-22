"""Move a recorded take to another take of the plan (it was recorded as something else).

    python -m capture.relabel data/river_samples/sat_follow2 54 57 --note "person waving ..."

The file is renamed to the target take, its attributes are replaced by the target take's
plan fields (marks and timing stay), and takes.csv is updated. The source take becomes
free again, so the guided capture offers it next time. Needs h5py.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import h5py

from capture.record import CSV_FIELDS, PLANS, session_plan


def relabel(session: Path, src: int, dst: int, note: str = "") -> Path:
    plan = PLANS[session_plan(session)]()
    if not 0 <= dst < len(plan):
        raise SystemExit(f"take {dst} is not in the plan (0-{len(plan) - 1})")
    index = session / "takes.csv"
    with index.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_take = {int(r["take"]): r for r in rows}
    if src not in by_take:
        raise SystemExit(f"take {src} is not recorded in {session}")
    if dst in by_take:
        raise SystemExit(f"take {dst} is already recorded in {session}; delete or move it first")
    take = plan[dst]
    old = session / by_take[src]["file"]
    new = session / f"take_{dst:03d}_{take.label}.h5"
    old.rename(new)
    with h5py.File(new, "r+") as f:
        for k, v in {"take": dst, **asdict(take)}.items():
            f.attrs[k] = v
        f.attrs["relabelled_from"] = f"take {src} ({by_take[src]['label']})" + (f": {note}" if note else "")
    r = by_take[src]
    r.update(take=str(dst), file=new.name, phase=str(take.phase), label=take.label, distance_m=take.distance_m,
             camera=take.camera, ball=take.ball, time_labelled=str(take.time_labelled), instruction=take.instruction)
    rows.sort(key=lambda r: int(r["take"]))
    with index.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"take {src} -> take {dst} ({take.label}): {new.name}")
    return new


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", type=Path)
    ap.add_argument("src", type=int, help="take number it was recorded as")
    ap.add_argument("dst", type=int, help="take number of the plan it really is")
    ap.add_argument("--note", default="", help="what was actually recorded")
    args = ap.parse_args()
    relabel(args.session, args.src, args.dst, args.note)


if __name__ == "__main__":
    main()
