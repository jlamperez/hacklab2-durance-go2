"""Event sources for the event-camera node: recordings played back, or the live camera.

Both hand out events in the repo's format (`EVENT_DTYPE`: t in µs, x, y, p) through the
same calls, so the node does not care where they come from:

    src.poll(now_us)    new events since the last call, sorted, all newer than before
    src.t_end(now_us)   stream time the next window should end at, or None (nothing yet)

`now_us` is a monotonic clock in microseconds supplied by the caller.

ReplaySource  `.h5` takes (data/river/README_recordings_*.md) or river `.bin` files: one
              file, or whole session folders, in real time (or at `speed`), looping.
LiveSource    the KV260 camera the way `river-tools capture` receives it: this side
              listens on a TCP port, `despot_debug_camera` on the board connects and sends
              batches (4-byte big-endian size, then 13-byte events: x >u2, y >u2,
              polarity u1, timestamp >i8 in µs). Read from river-tools 0.1.0
              (`capture/events.py`, `stream.py`, `remote.py`), not from a spec.

No ROS here: the node is thin, this is unit-tested (tests/test_sources.py).
"""
from __future__ import annotations

import csv
import fnmatch
import json
import socket
import struct
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .emu import EVENT_DTYPE, empty_events

RIVER_DTYPE = np.dtype([("x", ">u2"), ("y", ">u2"), ("p", "u1"), ("t", ">i8")])
EXTENSIONS = (".h5", ".bin")
INFO_KEYS = ("label", "instruction", "distance_m", "camera", "ball", "phase", "take")


def flip(ev: np.ndarray, width: int, height: int, flip_x: bool, flip_y: bool) -> np.ndarray:
    """Mirror the image (a camera mounted the other way round). In place, returns `ev`."""
    if flip_x:
        ev["x"] = (width - 1) - ev["x"].astype(np.int32)
    if flip_y:
        ev["y"] = (height - 1) - ev["y"].astype(np.int32)
    return ev


def read_take(path: Path) -> np.ndarray:
    """Events of one recording, sorted by time. h5py is imported only for `.h5` files."""
    from reflex.river.data import read_events
    return read_events(path)


def take_info(path: Path) -> dict:
    """What the recording says about itself (HDF5 root attributes). Empty for `.bin`."""
    if path.suffix != ".h5":
        return {}
    import h5py
    with h5py.File(path, "r") as f:
        out = {}
        for k in INFO_KEYS:
            if k in f.attrs:
                v = f.attrs[k]
                out[k] = v.decode() if isinstance(v, bytes) else (v.item() if hasattr(v, "item") else v)
        return out


# --------------------------------------------------------------------------- recordings

@dataclass
class Take:
    path: Path
    name: str                 # as shown in logs: relative to the folder it was found in
    info: dict = field(default_factory=dict)

    def describe(self) -> str:
        i = self.info
        bits = [str(i[k]) for k in ("label", "ball", "camera") if i.get(k) not in (None, "")]
        d = str(i.get("distance_m", "")).strip()
        if d:
            bits.append(f"{d} m" if d.replace(".", "", 1).isdigit() else d)
        return self.name + (f" ({', '.join(bits)})" if bits else "")


def _dropped(path: Path, root: Path) -> bool:
    """True if an overrides.json between `path` and `root` says {"drop": true} for it."""
    d = path.parent
    while True:
        ov = d / "overrides.json"
        if ov.exists():
            entry = json.loads(ov.read_text()).get(path.relative_to(d).as_posix(), {})
            if entry.get("drop"):
                return True
        if d == root or d == d.parent:
            return False
        d = d.parent


def _csv_order(session: Path) -> dict:
    """Row in takes.csv (recording order) by file name and by take number, if the session
    has one. The number matters for renamed files (none/empty_000.h5 was take_000)."""
    f = session / "takes.csv"
    if not f.exists():
        return {}
    order: dict = {}
    with f.open(newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if row.get("file"):
                order[row["file"]] = i
            if (row.get("take") or "").strip().isdigit():
                order[("take", int(row["take"]))] = i
    return order


def find_takes(paths: list[str | Path], exclude: list[str] | None = None,
               with_info: bool = True) -> list[Take]:
    """Every recording under `paths` (files or folders), in recording order.

    In a folder: takes listed in takes.csv follow its order, the rest by name. Takes that
    overrides.json drops are left out, and so are takes whose relative path or file name
    matches a pattern in `exclude` (fnmatch, e.g. "*take_002*"). When a `.bin` and its
    converted `.h5` both exist, the `.h5` is used (it carries the label).
    """
    exclude = exclude or []
    takes: list[Take] = []
    for p in map(Path, paths):
        if p.is_file():
            files, root = [p], p.parent
        elif p.is_dir():
            files = [f for f in p.rglob("*") if f.suffix in EXTENSIONS
                     and not (f.suffix == ".bin" and f.with_suffix(".h5").exists())]
            root = p
        else:
            raise FileNotFoundError(f"no recording or folder at {p}")
        found = []
        for f in files:
            rel = f.relative_to(root).as_posix() if f != p else f.name
            if any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(f.name, pat) for pat in exclude):
                continue
            if p.is_dir() and _dropped(f, root):
                continue
            found.append(Take(f, rel, take_info(f) if with_info else {}))

        def key(take):
            # session = the nearest folder with a takes.csv, else the take's own folder
            s = next((d for d in take.path.parents if (d / "takes.csv").exists()), take.path.parent)
            order = _csv_order(s)
            i = order.get(take.path.name, order.get(("take", take.info.get("take")), 1 << 30))
            return (str(s), i, take.name)

        takes += sorted(found, key=key)
    return takes


def session_of(path: Path) -> Path:
    """The session folder of a take: the nearest folder with a takes.csv, else the folder
    above its kind folder (<session>/<kind>/<take>)."""
    return next((d for d in path.parents if (d / "takes.csv").exists()), path.parent.parent)


def still_take(take: Take, with_info: bool = True) -> Take | None:
    """The session's recording of the still scene (`none/empty*`, the convention the
    autolabeller calibrates on), or None. Played first, it lets the target node and the
    simple detector calibrate on real sensor noise instead of on silence."""
    session = session_of(take.path)
    found = sorted(f for f in (session / "none").glob("empty*") if f.suffix in EXTENSIONS)
    if not found or found[0] == take.path:
        return None
    return Take(found[0], f"{session.name}/none/{found[0].name}", take_info(found[0]) if with_info else {})


class ReplaySource:
    """Plays recordings back in real time on one continuous timeline.

    Each take is shifted to start where the previous one ended plus `gap_s` of silence,
    so time only moves forward, also across loops. `lead_s` of silence comes first:
    the target node and the simple detector calibrate on the first seconds they see
    (play a session folder and its first take, the still scene, does that better).
    """

    def __init__(self, takes: list[Take], speed: float = 1.0, loop: bool = True,
                 gap_s: float = 1.0, lead_s: float = 0.0, width: int = 320, height: int = 320,
                 flip_x: bool = False, flip_y: bool = False,
                 reader: Callable[[Path], np.ndarray] = read_take):
        if not takes:
            raise ValueError("nothing to replay: no recordings found (or all excluded / dropped)")
        if speed <= 0:
            raise ValueError("speed must be > 0")
        self.takes, self.speed, self.loop = takes, speed, loop
        self.gap_us, self.width, self.height = int(gap_s * 1e6), width, height
        self.flip_x, self.flip_y, self.reader = flip_x, flip_y, reader
        self.loops = 0                  # completed passes through the list
        self._k = 0                     # next take to load
        self._ev: np.ndarray | None = None
        self._i = 0
        self._cur: Take | None = None
        self._start = self._end = 0     # timeline span of the loaded take
        self._next = int(lead_s * 1e6)  # timeline time the next take starts at
        self._t0_now: int | None = None
        self.finished = False

    def t_end(self, now_us: int) -> int:
        if self._t0_now is None:
            self._t0_now = now_us
        return int((now_us - self._t0_now) * self.speed)

    def current(self, now_us: int) -> tuple[Take, float] | None:
        """The take playing now and seconds into it, or None between takes."""
        t = self.t_end(now_us)
        if self._cur is None or self._ev is None or not self._start <= t <= self._end:
            return None
        return self._cur, (t - self._start) * 1e-6

    def _load_next(self) -> bool:
        for _ in range(len(self.takes)):
            if self._k >= len(self.takes):
                if not self.loop:
                    return False
                self._k, self.loops = 0, self.loops + 1
            take = self.takes[self._k]
            self._k += 1
            ev = self.reader(take.path)
            if not len(ev):
                continue
            ev = flip(ev.copy(), self.width, self.height, self.flip_x, self.flip_y)
            ev["t"] = ev["t"] - ev["t"][0] + self._next
            self._ev, self._i, self._cur = ev, 0, take
            self._start, self._end = int(ev["t"][0]), int(ev["t"][-1])
            return True
        return False           # every take is empty

    def poll(self, now_us: int) -> np.ndarray:
        t = self.t_end(now_us)
        chunks = []
        while not self.finished:
            if self._ev is None:
                if t < self._next:
                    break
                if not self._load_next():
                    self.finished = True
                    break
            i = int(np.searchsorted(self._ev["t"], t, side="right"))
            if i > self._i:
                chunks.append(self._ev[self._i:i])
                self._i = i
            if t < self._end:
                break
            self._next = self._end + self.gap_us
            self._ev = None
        return np.concatenate(chunks) if chunks else empty_events()


# --------------------------------------------------------------------------- live camera

class RiverStreamDecoder:
    """Bytes from despot_debug_camera's TCP stream -> events. Copes with any split of
    headers and events across reads (TCP gives no boundaries)."""

    def __init__(self):
        self._buf = bytearray()
        self._left = 0         # bytes of the current batch not decoded yet

    def feed(self, data: bytes) -> np.ndarray:
        self._buf += data
        out = []
        while True:
            if self._left == 0:
                if len(self._buf) < 4:
                    break
                self._left = struct.unpack("!I", self._buf[:4])[0]
                del self._buf[:4]
                if self._left % RIVER_DTYPE.itemsize:
                    raise ValueError(f"batch of {self._left} bytes is not a whole number of "
                                     f"{RIVER_DTYPE.itemsize}-byte events: not the river-tools protocol")
                continue
            n = min(self._left, len(self._buf)) // RIVER_DTYPE.itemsize * RIVER_DTYPE.itemsize
            if n == 0:
                break
            rec = np.frombuffer(bytes(self._buf[:n]), dtype=RIVER_DTYPE)
            del self._buf[:n]
            self._left -= n
            ev = np.empty(len(rec), dtype=EVENT_DTYPE)
            for k in ("t", "x", "y", "p"):
                ev[k] = rec[k]
            out.append(ev)
        return np.concatenate(out) if out else empty_events()


def start_camera_over_ssh(host: str, port: int) -> subprocess.Popen:
    """Start despot_debug_camera on the board, sending to this machine, as river-tools does
    (capture/remote.py): the board sees this machine's address in SSH_CONNECTION."""
    command = f"""set -- $SSH_CONNECTION
despot_debug_camera --host "$1" --port {port} &
camera_pid=$!
trap 'kill "$camera_pid" 2>/dev/null; wait "$camera_pid" 2>/dev/null' EXIT
trap 'exit' HUP INT TERM
read -r stop
"""
    return subprocess.Popen(["ssh", "-4", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
                             "-o", "ConnectTimeout=10", "--", host, command], stdin=subprocess.PIPE)


class LiveSource:
    """Listens on `port` for the camera's event stream.

    Camera clock -> stream time: the camera stamps events with its own clock. Windows must
    keep moving when the scene is still and nothing arrives, so the newest window ends at
    `now - offset`, offset = the smallest (arrival time - event time) seen so far, i.e.
    camera time minus the best-case network delay. A jump back in camera time by more
    than a second (camera restarted) starts over; `epoch` counts those, so the node can
    drop the old events.
    """

    def __init__(self, port: int = 3333, kria_host: str = "", width: int = 320, height: int = 320,
                 flip_x: bool = False, flip_y: bool = False,
                 clock: Callable[[], int] = lambda: time.monotonic_ns() // 1000,
                 log: Callable[[str], None] = print):
        self.port, self.kria_host = port, kria_host
        self.width, self.height, self.flip_x, self.flip_y = width, height, flip_x, flip_y
        self.clock, self.log = clock, log
        self.epoch = 0
        self.received = 0              # events received in total
        self.dropped_late = 0          # events older than ones already handed out
        self.connected = False
        self._lock = threading.Lock()
        self._pending: list[np.ndarray] = []
        self._offset: int | None = None
        self._last_cam_t: int | None = None
        self._last_out_t = -(1 << 62)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ssh: subprocess.Popen | None = None

    # -- called by the receiving thread (and by tests) --
    def ingest(self, ev: np.ndarray, arrival_us: int | None = None) -> None:
        if not len(ev):
            return
        arrival_us = self.clock() if arrival_us is None else arrival_us
        ev = flip(ev, self.width, self.height, self.flip_x, self.flip_y)
        newest = int(ev["t"].max())
        with self._lock:
            if self._last_cam_t is not None and newest < self._last_cam_t - 1_000_000:
                self.log("camera time jumped back (camera restarted?): starting over")
                self._pending, self._offset, self._last_out_t = [], None, -(1 << 62)
                self.epoch += 1
            self._last_cam_t = newest if self._last_cam_t is None else max(newest, self._last_cam_t)
            d = arrival_us - newest
            self._offset = d if self._offset is None else min(self._offset, d)
            self._pending.append(ev)
            self.received += len(ev)

    # -- called by the node --
    def t_end(self, now_us: int) -> int | None:
        with self._lock:
            return None if self._offset is None else int(now_us - self._offset)

    def poll(self, now_us: int) -> np.ndarray:
        with self._lock:
            chunks, self._pending = self._pending, []
            last = self._last_out_t
        if not chunks:
            return empty_events()
        ev = np.concatenate(chunks)
        if (np.diff(ev["t"]) < 0).any():
            ev = ev[np.argsort(ev["t"], kind="stable")]
        late = ev["t"] < last
        if late.any():
            self.dropped_late += int(late.sum())
            ev = ev[~late]
        if len(ev):
            with self._lock:
                self._last_out_t = max(self._last_out_t, int(ev["t"][-1]))
        return ev

    def start(self) -> None:
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._ssh is not None:
            try:
                self._ssh.communicate(input=b"\n", timeout=5)
            except subprocess.TimeoutExpired:
                self._ssh.kill()
                self._ssh.wait()

    def _serve(self) -> None:
        with socket.create_server(("", self.port)) as server:
            server.settimeout(0.2)
            self.log(f"waiting for the camera's events on TCP port {self.port}")
            if self.kria_host:
                self.log(f"starting despot_debug_camera on {self.kria_host} over ssh")
                self._ssh = start_camera_over_ssh(self.kria_host, self.port)
            while not self._stop.is_set():
                if self._ssh is not None and self._ssh.poll() is not None:
                    self.log("the camera's ssh session ended")
                    self._ssh = None
                try:
                    conn, addr = server.accept()
                except socket.timeout:
                    continue
                self.log(f"camera connected from {addr[0]}")
                decoder = RiverStreamDecoder()
                self.connected = True
                with conn:
                    conn.settimeout(0.2)
                    while not self._stop.is_set():
                        try:
                            data = conn.recv(1 << 16)
                        except socket.timeout:
                            continue
                        except OSError:
                            break
                        if not data:
                            break
                        try:
                            self.ingest(decoder.feed(data))
                        except ValueError as e:
                            self.log(f"dropping the connection: {e}")
                            break
                self.connected = False
                self.log("camera disconnected; waiting for it again")
