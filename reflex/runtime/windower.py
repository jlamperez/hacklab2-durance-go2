"""Keeps the most recent events of a live stream and cuts network windows from it.

Every live event source (simulator, webcam emulation, the real event camera) pushes
its events here and asks for a window when it wants a decision.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from reflex.data.windows import WindowConfig, make_window
from reflex.events.emu import empty_events


class StreamWindower:
    def __init__(self, cfg: WindowConfig | None = None):
        self.cfg = cfg or WindowConfig()
        self._chunks: deque[np.ndarray] = deque()

    def push(self, events: np.ndarray) -> None:
        """Add a chunk of events (sorted by time, newer than everything pushed before)."""
        if len(events):
            self._chunks.append(events)

    def recent(self, t_end_us: int, span_ms: float | None = None) -> np.ndarray:
        """Events of the last `span_ms` (default: one window). Older chunks are dropped."""
        t0 = t_end_us - int((span_ms or self.cfg.window_ms) * 1000)
        keep_from = t_end_us - int(self.cfg.window_ms * 1000)
        while self._chunks and self._chunks[0]["t"][-1] < min(t0, keep_from):
            self._chunks.popleft()
        if not self._chunks:
            return empty_events()
        ev = np.concatenate(self._chunks)
        return ev[ev["t"] >= t0]

    def window(self, t_end_us: int) -> np.ndarray:
        """Network input ending at `t_end_us`: uint8 (T, 2, H/ds, W/ds)."""
        return make_window(self.recent(t_end_us), t_end_us, self.cfg)
