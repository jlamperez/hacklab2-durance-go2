"""A zone name in, a `Target` out: for zone classifiers that only say which zone.

The FPGA ball node (`ball_node` in the colleague's `gesture_node` package, run on the
Kria next to the SPLEAT server) publishes the ball zone classifier's decision as a
string on /durance/ball_zone: one of `upper_left ... lower_right` or `none`
(train_ball_zones.CLASSES), majority-voted on the board and sent only when it changes.
The follow node wants a stream of `Target`s (u, v, confidence), so the `zone_target`
node keeps the latest zone and republishes it at a fixed rate through `target_from_zone`.

The zone becomes its centre, as in reflex/runtime/ball_snn.py: u in {-2/3, 0, +2/3},
v in {-1/2, +1/2}. The board sends no probability, so a zone gets a fixed confidence.

`target_from_decision` does the same for the three-zone network on the FPGA (models/zones3,
fpga_perception on /reflex/decision): left / center / right become u = -2/3, 0, +2/3.
"""
from __future__ import annotations

from reflex.runtime.target import Target, no_target

ZONES = ("upper_left", "upper_middle", "upper_right", "lower_left", "lower_middle", "lower_right")
ROWS, COLUMNS = 2, 3


def zone_center(index: int, rows: int = ROWS, columns: int = COLUMNS) -> tuple[float, float]:
    """(u, v) of a zone's centre, zones numbered row by row from the top left."""
    row, col = divmod(index, columns)
    return (col + 0.5) / columns * 2.0 - 1.0, (row + 0.5) / rows * 2.0 - 1.0


def target_from_zone(name: str, t_us: int = 0, confidence: float = 1.0) -> Target:
    """`none` (or an empty string) -> nothing seen; a zone -> its centre. Unknown names raise."""
    name = name.strip().lower()
    if name in ("", "none"):
        return no_target(t_us)
    if name not in ZONES:
        raise ValueError(f"unknown zone {name!r}, expected none or one of {', '.join(ZONES)}")
    u, v = zone_center(ZONES.index(name))
    return Target(u, v, confidence, t_us)


# Reflex decisions (reflex_msgs/ReflexDecision.label) name a third of the image, no row.
DECISION_U = {"left": -2.0 / 3.0, "center": 0.0, "right": 2.0 / 3.0}


def target_from_decision(label: str, confidence: float, t_us: int = 0, v: float = 0.0) -> Target:
    """A reflex decision (`none / left / center / right`, e.g. fpga_perception's zones3 readout)
    as a Target: the third's centre column, at image row `v` (0 = the middle, so look mode
    picks the level gestures left / center / right). `none` -> nothing seen. Unknown labels raise."""
    label = label.strip().lower()
    if label in ("", "none"):
        return no_target(t_us)
    if label not in DECISION_U:
        raise ValueError(f"unknown decision {label!r}, expected none, left, center or right")
    return Target(DECISION_U[label], v, confidence, t_us)
