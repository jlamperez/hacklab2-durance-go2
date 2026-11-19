"""Recording real data and turning it into a training set.

A recording session is a folder of clips. One clip is one take: a ball rolled at the dog
(label left / center / right) or something that must NOT trigger a reflex (label none:
someone walking past, a ball crossing, the dog just standing).

For a ball, the operator marks the moment it arrives ("hit"). Labels are derived from
that mark when the dataset is built, not when recording, so the rule can be changed
later without recording again:

      ... none ... | discarded | threat (label) | discarded
                   ^           ^                ^ hit
                   hit - threat_ms - guard_ms   hit - threat_ms

The discarded stretch before the threat is the ambiguous part: the ball is on its way
but still far.

    data/real/<session>/clip_000.npz    windows uint8 (N, T, 2, H, W), t_us int64 (N,),
                                        label, t_hit_us (-1 for "none"), window config
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LABELS = ("none", "left", "center", "right")
KINDS = ("approach", "negative")
DISCARD = -1


def label_clip(t_us: np.ndarray, label: str, t_hit_us: int, threat_ms: float = 400.0,
               guard_ms: float = 300.0) -> np.ndarray:
    """Label index for each window of a clip, DISCARD where it would be ambiguous."""
    t_us = np.asarray(t_us, dtype=np.int64)
    if label == "none":
        return np.zeros(len(t_us), dtype=np.int64)
    if t_hit_us < 0:
        raise ValueError("a threat clip needs the time the ball arrived")
    start = t_hit_us - int(threat_ms * 1000)
    y = np.zeros(len(t_us), dtype=np.int64)
    y[(t_us >= start - int(guard_ms * 1000)) & (t_us < start)] = DISCARD
    y[(t_us >= start) & (t_us <= t_hit_us)] = LABELS.index(label)
    y[t_us > t_hit_us] = DISCARD
    return y


class ClipRecorder:
    """Collects the windows of one clip at a time and writes them into a session folder."""

    def __init__(self, session_dir: str | Path, window_cfg: dict | None = None):
        self.dir = Path(session_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.window_cfg = window_cfg or {}
        self.label: str | None = None
        self._windows: list[np.ndarray] = []
        self._t: list[int] = []
        self.saved: list[Path] = sorted(self.dir.glob("clip_*.npz"))

    @property
    def recording(self) -> bool:
        return self.label is not None

    def start(self, label: str) -> None:
        if label not in LABELS:
            raise ValueError(f"unknown label {label!r}")
        self.label, self._windows, self._t = label, [], []

    def add(self, window: np.ndarray, t_us: int) -> None:
        if self.recording:
            self._windows.append(np.asarray(window, dtype=np.uint8))
            self._t.append(int(t_us))

    def stop(self, t_hit_us: int | None = None) -> Path | None:
        """End the clip. For a threat clip `t_hit_us` is when the ball arrived."""
        label, self.label = self.label, None
        if label is None or not self._windows:
            return None
        if label != "none" and t_hit_us is None:
            t_hit_us = self._t[-1]
        n = int(self.saved[-1].stem.split("_")[1]) + 1 if self.saved else 0
        path = self.dir / f"clip_{n:03d}.npz"
        np.savez_compressed(path, windows=np.stack(self._windows), t_us=np.array(self._t, dtype=np.int64),
                            label=label, t_hit_us=-1 if label == "none" else int(t_hit_us),
                            window_cfg=json.dumps(self.window_cfg))
        self.saved.append(path)
        return path

    def cancel(self) -> None:
        self.label = None

    def undo(self) -> Path | None:
        """Delete the last saved clip (a bad take)."""
        if not self.saved:
            return None
        path = self.saved.pop()
        path.unlink(missing_ok=True)
        return path

    def counts(self) -> dict:
        out = {lab: 0 for lab in LABELS}
        for p in self.saved:
            out[str(np.load(p)["label"])] += 1
        return out


def build_dataset(session_dirs, out_dir: str | Path, threat_ms: float = 400.0, guard_ms: float = 300.0) -> dict:
    """Turn recorded sessions into the shard format `reflex.train` reads.

    One shard per session. `episode` is a clip id unique across sessions, so the
    train/validation split never puts windows of one take on both sides; `session` lets a
    whole session be held out.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("shard_*.npz"):
        old.unlink()
    sessions, clip_id, total, window_cfg = [], 0, np.zeros(len(LABELS), dtype=int), None
    for s_idx, sdir in enumerate(sorted(Path(d) for d in session_dirs)):
        xs, ys, eps, kinds = [], [], [], []
        for clip in sorted(sdir.glob("clip_*.npz")):
            c = np.load(clip)
            label = str(c["label"])
            y = label_clip(c["t_us"], label, int(c["t_hit_us"]), threat_ms, guard_ms)
            keep = y != DISCARD
            xs.append(c["windows"][keep])
            ys.append(y[keep])
            eps.append(np.full(int(keep.sum()), clip_id))
            kinds.append(np.full(int(keep.sum()), KINDS.index("negative" if label == "none" else "approach")))
            window_cfg = window_cfg or json.loads(str(c["window_cfg"]))
            clip_id += 1
        if not xs:
            continue
        x, y = np.concatenate(xs), np.concatenate(ys)
        np.savez_compressed(out_dir / f"shard_{s_idx:03d}.npz", x=x, y=y, episode=np.concatenate(eps),
                            kind=np.concatenate(kinds), session=np.full(len(y), len(sessions)))
        sessions.append(sdir.name)
        total += np.bincount(y, minlength=len(LABELS))
    if not sessions:
        raise SystemExit("no clips found")
    meta = {"labels": list(LABELS), "kinds": list(KINDS), "window": window_cfg, "sessions": sessions,
            "episodes": clip_id, "windows": int(total.sum()), "threat_ms": threat_ms, "guard_ms": guard_ms,
            "source": "recorded sessions: " + ", ".join(sessions)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return {"sessions": sessions, "clips": clip_id, "windows": dict(zip(LABELS, total.tolist()))}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Build a training set from recorded sessions.")
    ap.add_argument("sessions", nargs="+", help="session folders, e.g. data/real/*")
    ap.add_argument("--out", default="data/real_ds")
    ap.add_argument("--threat-ms", type=float, default=400.0, help="how long before arrival a ball counts as a threat")
    ap.add_argument("--guard-ms", type=float, default=300.0, help="ambiguous stretch before that, left out")
    args = ap.parse_args()
    print(json.dumps(build_dataset(args.sessions, args.out, args.threat_ms, args.guard_ms), indent=2))
    print(f"wrote {args.out}; look at it with: pixi run view-data {args.out}")


if __name__ == "__main__":
    main()
