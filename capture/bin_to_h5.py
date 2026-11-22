"""Convert river-tools .bin recordings (and older guided-capture takes) to HDF5.

    python -m capture.bin_to_h5 data/river_samples/samples          # every .bin in the folder
    python -m capture.bin_to_h5 data/river_samples/sat_follow1      # a session: also updates takes.csv
    python -m capture.bin_to_h5 some_take.bin --remove              # delete the .bin (and .json) once verified

Each X.bin becomes X.h5 next to it (layout in reflex/events/h5.py). A take's X.json, if
present, goes into the attributes. Every file is read back and compared with the original
before anything is removed; without --remove the originals stay.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from reflex.events.h5 import CAMERA_SIZE, read_bin, read_h5, write_h5


def convert(path: Path, remove: bool) -> Path:
    ev = read_bin(path)
    meta = {"source_file": path.name}
    side = path.with_suffix(".json")
    if side.exists():
        meta.update(json.loads(side.read_text(encoding="utf-8")))
        meta.pop("format", None)          # describes the old .bin, not the new file
    out = write_h5(path.with_suffix(".h5"), ev, meta, CAMERA_SIZE, CAMERA_SIZE)
    t, x, y, p, _ = read_h5(out)
    same = (len(t) == len(ev) and np.array_equal(t, ev["timestamp"]) and np.array_equal(x, ev["x"])
            and np.array_equal(y, ev["y"]) and np.array_equal(p, ev["polarity"]))
    if not same:
        out.unlink()
        raise RuntimeError(f"{out.name}: read-back differs from {path.name}; originals kept")
    if remove:
        path.unlink()
        side.unlink(missing_ok=True)
    mb = lambda q: q.stat().st_size / 2**20
    print(f"{path.name} -> {out.name}: {len(ev)} events, {mb(out):.1f} MB")
    return out


def update_index(folder: Path) -> None:
    """Point takes.csv at the .h5 files that now exist."""
    index = folder / "takes.csv"
    if not index.exists():
        return
    with index.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    for r in rows:
        h5 = Path(r["file"]).with_suffix(".h5").name
        if (folder / h5).exists():
            r["file"] = h5
    with index.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"updated {index}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help=".bin files or folders containing them")
    ap.add_argument("--remove", action="store_true", help="delete each .bin (and .json) after a verified conversion")
    args = ap.parse_args()
    for item in map(Path, args.paths):
        files = sorted(item.glob("*.bin")) if item.is_dir() else [item]
        for f in files:
            convert(f, args.remove)
        update_index(item if item.is_dir() else item.parent)


if __name__ == "__main__":
    main()
