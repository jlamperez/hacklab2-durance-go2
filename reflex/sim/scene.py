"""MuJoCo scene: a standing Go2, a ball launcher and a head-mounted camera.

Scope, on purpose: the simulated dog does not walk. On the real robot, walking is
handled by Unitree's sport mode. Here the dog holds a pose with a joint PD
controller and reflexes are posture changes. The simulation exists to generate
event data and to test the closed loop, nothing else.

Frame convention (dog at the origin): +x forward, +y left, +z up.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
GO2_XML = REPO_ROOT / "third_party" / "mujoco_menagerie" / "unitree_go2" / "go2.xml"

# Joint targets per leg: (abduction, hip, knee). Leg order: FL, FR, RL, RR.
POSES: dict[str, np.ndarray] = {
    "stand": np.tile([0.0, 0.80, -1.50], 4),
    "crouch": np.tile([0.0, 1.25, -2.40], 4),
    # Leaning shifts the trunk sideways by rolling all four abduction joints.
    "lean_left": np.tile([-0.35, 0.80, -1.50], 4),
    "lean_right": np.tile([0.35, 0.80, -1.50], 4),
}

KP, KD = 60.0, 3.0
LABELS = ("none", "left", "center", "right")


@dataclass
class BallShot:
    """One ball rolled at the dog."""
    label: str          # where it arrives, in the dog frame: left / center / right / none
    radius: float       # m
    speed: float        # m/s
    x_start: float
    y_start: float
    y_target: float     # lateral position when it reaches the dog
    rgb: tuple[float, float, float]

    @classmethod
    def sample(cls, rng: np.random.Generator, label: str | None = None) -> "BallShot":
        label = label or str(rng.choice(LABELS[1:]))
        y_target = {
            "left": rng.uniform(0.22, 0.55),
            "center": rng.uniform(-0.12, 0.12),
            "right": rng.uniform(-0.55, -0.22),
            # A miss: passes well to one side.
            "none": rng.choice([-1, 1]) * rng.uniform(1.3, 2.5),
        }[label]
        return cls(
            label=label,
            radius=float(rng.uniform(0.05, 0.12)),
            speed=float(rng.uniform(2.0, 6.0)),
            x_start=float(rng.uniform(3.0, 5.0)),
            y_start=float(rng.uniform(-1.2, 1.2)),
            y_target=float(y_target),
            rgb=tuple(float(c) for c in rng.uniform(0.15, 1.0, size=3)),
        )


def _look_at(pos, target) -> list[float]:
    """MuJoCo camera `xyaxes` (right, up) for a camera at `pos` looking at `target`."""
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    right = np.cross(f, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    return [*right, *up]


def build_model(rng: np.random.Generator | None = None) -> mujoco.MjModel:
    if not GO2_XML.exists():
        raise FileNotFoundError(f"{GO2_XML} not found. Run `pixi run setup` to fetch the Go2 model.")
    rng = rng or np.random.default_rng(0)
    G = mujoco.mjtGeom
    spec = mujoco.MjSpec.from_file(str(GO2_XML))
    for key in list(spec.keys):   # the stock keyframe no longer matches once the ball is added
        spec.delete(key)

    def checker(name, rgb1, rgb2, repeat):
        spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D,
                         builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                         rgb1=rgb1, rgb2=rgb2, width=256, height=256)
        mat = spec.add_material(name=name, texrepeat=repeat, texuniform=True)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = name

    checker("floor", [0.55, 0.55, 0.52], [0.42, 0.43, 0.42], [2, 2])
    checker("wall", [0.75, 0.72, 0.66], [0.55, 0.56, 0.60], [1.5, 1.5])
    checker("ball", [1.0, 1.0, 1.0], [0.25, 0.25, 0.25], [4, 4])   # panels, tinted by rgba
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.62, 0.70, 0.80], rgb2=[0.30, 0.34, 0.40], width=256, height=1536)

    spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
    spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
    spec.visual.headlight.specular = [0, 0, 0]
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720

    w = spec.worldbody
    w.add_light(pos=[1.5, 0, 4], dir=[0, 0, -1], diffuse=[0.6, 0.6, 0.6])
    w.add_geom(name="floor", type=G.mjGEOM_PLANE, size=[0, 0, 0.05], material="floor")
    w.add_geom(name="back_wall", type=G.mjGEOM_BOX, pos=[7.0, 0, 1.0], size=[0.05, 5, 1.0], material="wall")
    # Static clutter so the background is not empty (matters once the camera shakes).
    for i in range(6):
        w.add_geom(name=f"clutter{i}", type=G.mjGEOM_BOX,
                   pos=[rng.uniform(5.0, 6.6), rng.uniform(-4, 4), 0.0],
                   size=[rng.uniform(0.1, 0.3), rng.uniform(0.1, 0.4), rng.uniform(0.2, 0.9)],
                   rgba=[*rng.uniform(0.1, 0.9, size=3), 1.0])

    ball = w.add_body(name="ball", pos=[20, 0, 0.1])
    ball.add_freejoint()
    ball.add_geom(name="ball", type=G.mjGEOM_SPHERE, size=[0.1, 0, 0], mass=0.3,
                  material="ball", rgba=[1.0, 0.45, 0.1, 1.0], friction=[0.8, 0.005, 0.0001])

    # Event camera on the head: looks along +x.
    spec.body("base").add_camera(name="eye", pos=[0.33, 0.0, 0.05],
                                 xyaxes=[0, -1, 0, 0, 0, 1], fovy=60)
    third_pos = [-1.1, -1.9, 0.95]
    w.add_camera(name="third", pos=third_pos, xyaxes=_look_at(third_pos, [1.3, 0.1, 0.15]), fovy=50)
    return spec.compile()


class Go2Sim:
    """Standing Go2 with a PD pose controller and a ball launcher."""

    def __init__(self, seed: int = 0, eye_hw: tuple[int, int] = (180, 320),
                 third_hw: tuple[int, int] = (360, 640)):
        self.rng = np.random.default_rng(seed)
        self.model = build_model(self.rng)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.dt = m.opt.timestep
        self._ball_geom = m.geom("ball").id
        self._ball_qadr = m.joint(m.body("ball").jntadr[0]).qposadr[0]
        self._ball_vadr = m.joint(m.body("ball").jntadr[0]).dofadr[0]
        jnt = m.actuator_trnid[:, 0]
        self._q_adr = m.jnt_qposadr[jnt]
        self._v_adr = m.jnt_dofadr[jnt]
        self._ctrl_lo, self._ctrl_hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
        self.q_des = POSES["stand"].copy()
        # Offscreen renderers are created on first use, so the interactive viewer
        # (which owns its own window and GL context) never has to create them.
        self._eye: mujoco.Renderer | None = None
        self._third: mujoco.Renderer | None = None
        self.eye_hw, self.third_hw = eye_hw, third_hw
        self.reset()

    @property
    def time(self) -> float:
        return float(self.data.time)

    def reset(self, settle_s: float = 0.6) -> None:
        d = self.data
        mujoco.mj_resetData(self.model, d)
        d.qpos[0:3] = [0, 0, 0.36]
        d.qpos[3:7] = [1, 0, 0, 0]
        d.qpos[self._q_adr] = POSES["stand"]
        self.q_des = POSES["stand"].copy()
        self.park_ball()
        for _ in range(int(settle_s / self.dt)):
            self.step()
        d.time = 0.0

    def set_pose(self, name: str) -> None:
        self.q_des = POSES[name].copy()

    def park_ball(self) -> None:
        a, v = self._ball_qadr, self._ball_vadr
        self.data.qpos[a:a + 7] = [20, 0, 0.2, 1, 0, 0, 0]
        self.data.qvel[v:v + 6] = 0

    def launch(self, shot: BallShot) -> None:
        m, d = self.model, self.data
        m.geom_size[self._ball_geom, 0] = shot.radius
        m.geom_rbound[self._ball_geom] = shot.radius
        m.geom_rgba[self._ball_geom, :3] = shot.rgb
        start = np.array([shot.x_start, shot.y_start])
        direction = np.array([0.35, shot.y_target]) - start
        direction /= np.linalg.norm(direction)
        vel = shot.speed * direction
        a, v = self._ball_qadr, self._ball_vadr
        d.qpos[a:a + 7] = [start[0], start[1], shot.radius, 1, 0, 0, 0]
        # Pure rolling: omega = z x v / r
        d.qvel[v:v + 6] = [vel[0], vel[1], 0, -vel[1] / shot.radius, vel[0] / shot.radius, 0]

    def ball_xy(self) -> np.ndarray:
        return self.data.qpos[self._ball_qadr:self._ball_qadr + 2].copy()

    def step(self) -> None:
        d = self.data
        tau = KP * (self.q_des - d.qpos[self._q_adr]) - KD * d.qvel[self._v_adr]
        d.ctrl[:] = np.clip(tau, self._ctrl_lo, self._ctrl_hi)
        mujoco.mj_step(self.model, d)

    def render_eye(self) -> np.ndarray:
        """RGB uint8 (H, W, 3) from the head camera."""
        if self._eye is None:
            self._eye = mujoco.Renderer(self.model, *self.eye_hw)
        self._eye.update_scene(self.data, camera="eye")
        return self._eye.render()

    def render_third(self) -> np.ndarray:
        if self._third is None:
            self._third = mujoco.Renderer(self.model, *self.third_hw)
        self._third.update_scene(self.data, camera="third")
        return self._third.render()


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 -> luminance float32 in [0, 1]."""
    return (rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)) / 255.0
