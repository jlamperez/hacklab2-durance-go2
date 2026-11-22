"""Majority vote over the last few zone decisions, as the Kria's ball_node does before it
publishes on /durance/ball_zone.

The real ball_node (colleague's `gesture_node` package) majority-votes the zones network's
decisions on the board and sends a zone only when it changes. Its window length and tie rule
are not known to us; here both are config: the last `window` decisions are kept, a zone
(or `none`) wins when it holds at least `min_votes` of them, otherwise the current zone stays.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

from reflex.runtime.zone_target import ZONES

NAMES = ZONES + ("none",)


@dataclass
class ZoneVoteConfig:
    window: int = 5            # decisions kept
    min_votes: int = 3         # a zone needs this many of them to take over


class ZoneVoter:
    def __init__(self, cfg: ZoneVoteConfig | None = None):
        self.cfg = cfg = cfg or ZoneVoteConfig()
        if not 1 <= cfg.min_votes <= cfg.window:
            raise ValueError(f"need 1 <= min_votes ({cfg.min_votes}) <= window ({cfg.window})")
        self.history: deque[str] = deque(maxlen=cfg.window)
        self.current = "none"

    def update(self, zone: str) -> str | None:
        """Add one decision. Returns the new zone when the vote changes it, else None."""
        zone = zone.strip().lower() or "none"
        if zone not in NAMES:
            raise ValueError(f"unknown zone {zone!r}, expected one of {', '.join(NAMES)}")
        self.history.append(zone)
        winner, votes = Counter(self.history).most_common(1)[0]
        if votes >= self.cfg.min_votes and winner != self.current:
            self.current = winner
            return winner
        return None
