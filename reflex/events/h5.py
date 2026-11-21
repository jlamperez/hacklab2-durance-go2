"""HDF5 files for recorded event-camera takes, and reading river-tools' .bin format.

One take per file:

    /events/t   int64   microseconds, camera clock, sorted
    /events/x   uint16  0..width-1
    /events/y   uint16  0..height-1
    /events/p   uint8   polarity as sent by the camera (0 / 1)
    attributes on the root: format, width, height, t_unit, polarity, and the take's metadata
    (label, instruction, t_start_us, t_end_us, marks_us, ...)

Field names match the repo's EVENT_DTYPE (t, x, y, p). Which polarity value means "brighter"
is not documented by Durance; river-tools only colours 0 and 1 differently.

Plain HDF5 with gzip, readable with h5py, MATLAB, HDFView. Not Prophesee's HDF5 event format,
which needs their compression plugin.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

FORMAT = "reflex-events-v1"
CAMERA_SIZE = 320

# river-tools 0.1.0 wire and file format: uint32 BE payload size, then these records.
RIVER_DTYPE = np.dtype([("x", ">u2"), ("y", ">u2"), ("polarity", "u1"), ("timestamp", ">i8")])


def read_bin(path: str | Path) -> np.ndarray:
    """Events of a river-tools .bin file (RIVER_DTYPE). Tolerates a header left at 0."""
    raw = Path(path).read_bytes()
    size = int.from_bytes(raw[:4], "big")
    body = raw[4:]
    if not (0 < size <= len(body)):
        size = len(body)
    return np.frombuffer(body[: size - size % RIVER_DTYPE.itemsize], dtype=RIVER_DTYPE)


def write_bin(path: str | Path, t, x, y, p) -> Path:
    """Write events in river-tools' .bin format (readable by `river-tools play` and Durance's notebook)."""
    ev = np.zeros(len(t), dtype=RIVER_DTYPE)
    ev["x"], ev["y"], ev["polarity"], ev["timestamp"] = x, y, p, t
    data = ev.tobytes()
    path = Path(path)
    with path.open("wb") as f:
        f.write(len(data).to_bytes(4, "big"))
        f.write(data)
    return path


def _attr(value):
    """Attribute value h5py can store, or None to skip it."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.int64)
    if isinstance(value, (bool, int, float, str, np.integer, np.floating)):
        return value
    return str(value)


def write_h5(path: str | Path, events: np.ndarray, attrs: dict | None = None,
             width: int = CAMERA_SIZE, height: int = CAMERA_SIZE) -> Path:
    """Write events (RIVER_DTYPE, or any array with x, y, polarity/p, timestamp/t) to HDF5."""
    import h5py

    names = events.dtype.names
    t = np.asarray(events["timestamp" if "timestamp" in names else "t"], dtype=np.int64)
    p = np.asarray(events["polarity" if "polarity" in names else "p"], dtype=np.uint8)
    x = np.asarray(events["x"], dtype=np.uint16)
    y = np.asarray(events["y"], dtype=np.uint16)
    path = Path(path)
    tmp = path.with_suffix(".h5.tmp")
    opts = dict(compression="gzip", compression_opts=4, shuffle=True) if len(t) else {}
    with h5py.File(tmp, "w") as f:
        g = f.create_group("events")
        for name, data in (("t", t), ("x", x), ("y", y), ("p", p)):
            g.create_dataset(name, data=data, chunks=True if len(t) else None, **opts)
        f.attrs.update({"format": FORMAT, "width": width, "height": height, "t_unit": "us",
                        "polarity": "0 / 1 as sent by the camera; ON/OFF meaning unverified",
                        "n_events": len(t)})
        for k, v in (attrs or {}).items():
            v = _attr(v)
            if v is not None:
                f.attrs[k] = v
    tmp.replace(path)
    return path


def read_h5(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """(t, x, y, p, attrs) of a take written by write_h5."""
    import h5py

    with h5py.File(path, "r") as f:
        g = f["events"]
        t, x, y, p = (g[k][()] for k in ("t", "x", "y", "p"))
        attrs = {}
        for k, v in f.attrs.items():
            if isinstance(v, bytes):
                v = v.decode()
            elif isinstance(v, np.ndarray):
                v = v.tolist()
            elif isinstance(v, np.generic):
                v = v.item()
            attrs[k] = v
    return t, x, y, p, attrs
