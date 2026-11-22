"""Link the capture sessions (capture/, takes.csv per session) into the layout reflex.river
expects: <out>/<session>/<kind>/*.h5, kind `ball` (labelled by position by the auto-labeller)
or `none` (no target), plus `none/empty_*.h5` (still scene) for the noise calibration.

The 21 Nov recordings are one room, one camera position, one evening, so "sessions" here
are splits, not places: every take goes to `train` or `val` by its take number, and every
split gets the same still-scene takes for calibration. Symlinks only, nothing is copied.

    python tools/river_layout.py data/recordings_2026-11-21* --out data/river
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

BALL = {"from-left", "from-right", "from-center", "to-left", "to-right"}
NONE = {"no-ball", "none"}
# sat_follow2 takes 15 and 16: still scene, 20-40 events/s of noise (docs/recording-real-data.md).
EMPTY = {("sat_follow2", 15), ("sat_follow2", 16)}


def split_of(take: int, val_every: int) -> str:
    return "val" if take % val_every == val_every - 1 else "train"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", type=Path, help="folders holding session folders with takes.csv")
    ap.add_argument("--out", type=Path, default=Path("data/river"))
    ap.add_argument("--val-every", type=int, default=5, help="every n-th take (by number) goes to val")
    args = ap.parse_args()

    counts: dict[tuple[str, str], int] = {}
    empties = []
    for csv_path in sorted(p for r in args.roots for p in r.rglob("takes.csv")):
        session = csv_path.parent
        for row in csv.DictReader(csv_path.open()):
            take, label = int(row["take"]), row["label"]
            src = (session / row["file"]).resolve()
            if not src.exists():
                continue
            if (session.name, take) in EMPTY:
                empties.append(src)
                continue
            kind = "ball" if label in BALL else "none" if label in NONE else None
            if kind is None:
                continue
            split = split_of(take, args.val_every)
            dst = args.out / split / kind / f"{session.name}_{src.name}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src)
            counts[split, kind] = counts.get((split, kind), 0) + 1

    for split in ("train", "val"):
        for i, src in enumerate(empties):
            dst = args.out / split / "none" / f"empty_{i:03d}.h5"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src)
    for (split, kind), n in sorted(counts.items()):
        print(f"{split:5s} {kind:4s} {n:3d} takes")
    print(f"still-scene takes for calibration, in every split: {len(empties)}")


if __name__ == "__main__":
    main()
