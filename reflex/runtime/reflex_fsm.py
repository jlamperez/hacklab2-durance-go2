"""Reflex state machine: from a stream of decisions to at most one action at a time.

    idle --(N agreeing threats in a row)--> reacting --(hold)--> refractory --(rest)--> idle

It is the same for the SNN and for the simple detector, in simulation and on the robot.
Its job is to keep a twitchy detector from making the dog twitch:
  - a threat must be seen `consecutive` times in a row, with enough confidence
  - while reacting or resting, new decisions are ignored (the dog's own motion floods
    the camera with events, so anything seen then is not to be trusted)
"""
from __future__ import annotations

from dataclasses import dataclass

from reflex.runtime.detectors import Decision

# What to do for each label. Pose names match reflex.sim.scene.POSES; on the real robot
# the behavior node maps them to sport-mode commands.
BEHAVIORS = {
    "keeper": {"left": "lean_left", "center": "crouch", "right": "lean_right"},   # go to the ball
    "dodge": {"left": "lean_right", "center": "crouch", "right": "lean_left"},    # get out of the way
}


@dataclass
class FSMConfig:
    mode: str = "keeper"
    min_confidence: float = 0.6
    consecutive: int = 2          # agreeing threat decisions needed to trigger
    hold_s: float = 0.6           # how long the reaction pose is held
    refractory_s: float = 0.8     # rest after returning to stand


@dataclass(frozen=True)
class Action:
    name: str          # a pose: lean_left / lean_right / crouch / stand
    label: str         # the label that caused it ("none" for the return to stand)
    t_s: float


class ReflexFSM:
    def __init__(self, cfg: FSMConfig | None = None):
        self.cfg = cfg or FSMConfig()
        if self.cfg.mode not in BEHAVIORS:
            raise ValueError(f"unknown mode {self.cfg.mode!r}")
        self.state = "idle"
        self._streak_label, self._streak = "none", 0
        self._t_state = 0.0

    def update(self, decision: Decision | None, t_s: float) -> Action | None:
        """Feed one decision (or None just to advance time). Returns an action or None."""
        c = self.cfg
        if self.state == "reacting":
            if t_s - self._t_state >= c.hold_s:
                self.state, self._t_state = "refractory", t_s
                return Action("stand", "none", t_s)
            return None
        if self.state == "refractory":
            if t_s - self._t_state >= c.refractory_s:
                self.state, self._streak = "idle", 0
            return None
        if decision is None:
            return None
        if decision.is_threat and decision.confidence >= c.min_confidence:
            self._streak = self._streak + 1 if decision.label == self._streak_label else 1
            self._streak_label = decision.label
        else:
            self._streak, self._streak_label = 0, "none"
        if self._streak >= c.consecutive:
            self.state, self._t_state, self._streak = "reacting", t_s, 0
            return Action(BEHAVIORS[c.mode][decision.label], decision.label, t_s)
        return None
