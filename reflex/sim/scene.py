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
    # Leaning rolls the trunk in place, like the real Go2's Euler lean (Pose + Euler x=0.3,
    # measured 21 Nov 2026: 8 degrees towards that side, trunk moved under 1 cm). The legs on
    # that side bend 5 % of the way to crouch, the others extend as much, and the abduction
    # joints hold the trunk over its feet: 8.6 degrees, no sideways shift.
    "lean_left": np.array([0.15, 0.822, -1.545, 0.15, 0.778, -1.455, 0.15, 0.822, -1.545, 0.15, 0.778, -1.455]),
    "lean_right": np.array([-0.15, 0.778, -1.455, -0.15, 0.822, -1.545, -0.15, 0.778, -1.455, -0.15, 0.822, -1.545]),
}

# Slow, statically stable tricks, found by trying joint angles in the simulator. They are
# our own imitation: Unitree's Sit and Hello on the real robot look and time differently.
_SIT = np.array([0, 0.75, -1.2, 0, 0.75, -1.2, 0, 2.2, -2.6, 0, 2.2, -2.6], dtype=float)
# Weight on the front-left leg and the rump, front-right paw up.
_PAW_UP = np.array([-0.3, 0.95, -1.2, 0.35, -1.2, -1.3, 0.35, 2.2, -2.6, 0.35, 2.2, -2.6], dtype=float)
_PAW_WAVE = _PAW_UP.copy()
_PAW_WAVE[5] = -2.0
# name -> keyframes (seconds to reach it, joint targets). The last keyframe is held.
MOTIONS = {
    "sit": [(0.8, _SIT)],
    "rise": [(0.8, POSES["stand"])],
    "hello": [(0.8, _SIT), (0.6, _PAW_UP), (0.25, _PAW_WAVE), (0.25, _PAW_UP), (0.25, _PAW_WAVE),
              (0.25, _PAW_UP), (0.25, _PAW_WAVE), (0.25, _PAW_UP), (0.6, _SIT), (0.8, POSES["stand"])],
}

KP, KD = 60.0, 3.0

# Sidestep without the walking policy (Go2Sim.set_slide): the standing dog is moved as a
# whole, its speed following the command as a first-order lag. Fitted to a Move y=0.3 for
# 1 s on the real Go2 (21 Nov 2026): 0.4 / 1.4 / 3.0 / 6.9 / 20.0 cm at 0.1 / 0.2 / 0.3 /
# 0.5 / 1.0 s, 21.6 cm in total. Speeding up takes 0.37 s, stopping 0.06 s.
SLIDE_TAU_UP, SLIDE_TAU_DOWN = 0.37, 0.06

# Eachine UZ65-like tiny whoop (approximate): motor offset per axis for a 65 mm diagonal,
# duct radius just outside the 31 mm props.
WHOOP_ARM = 0.065 / 2 / np.sqrt(2)
WHOOP_DUCT_R = 0.0175
WHOOP_MOTORS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
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


def build_model(rng: np.random.Generator | None = None, scene: str = "lab") -> mujoco.MjModel:
    """`scene`: "lab" (default: checkered floor, wall, boxes) or "park" (reflex/sim/scenery.py)."""
    from reflex.sim import scenery
    if scene not in scenery.SCENES:
        raise ValueError(f"unknown scene {scene!r}; choose one of {scenery.SCENES}")
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

    checker("ball", [1.0, 1.0, 1.0], [0.25, 0.25, 0.25], [4, 4])   # panels, tinted by rgba

    spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
    spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
    spec.visual.headlight.specular = [0, 0, 0]
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720

    w = spec.worldbody
    if scene == "park":
        scenery.add_park(spec, rng)
    else:
        checker("floor", [0.55, 0.55, 0.52], [0.42, 0.43, 0.42], [2, 2])
        checker("wall", [0.75, 0.72, 0.66], [0.55, 0.56, 0.60], [1.5, 1.5])
        spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                         builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                         rgb1=[0.62, 0.70, 0.80], rgb2=[0.30, 0.34, 0.40], width=256, height=1536)
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

    # A person-sized shape moved kinematically, used as a hard negative ("someone walks by").
    walker = w.add_body(name="walker", mocap=True, pos=[30, 0, 0])
    for name, gtype, pos, size in (
        ("torso", G.mjGEOM_BOX, [0, 0, 1.15], [0.11, 0.20, 0.32]),
        ("leg_a", G.mjGEOM_CAPSULE, [0.06, 0.10, 0.42], [0.07, 0.40, 0]),
        ("leg_b", G.mjGEOM_CAPSULE, [-0.06, -0.10, 0.42], [0.07, 0.40, 0]),
    ):
        walker.add_geom(name=f"walker_{name}", type=gtype, pos=pos, size=size,
                        rgba=[0.2, 0.25, 0.5, 1.0], contype=0, conaffinity=0)

    # Tiny-whoop drone sized like an Eachine UZ65 (approximate public specs, not measured):
    # 65 mm motor-to-motor diagonal, 31 mm two-blade props inside ducts, about 85 mm across
    # and 30 mm high. Moved kinematically. The frame (pod + ducts) and the four props are
    # separate mocap bodies so the props can spin.
    drone = w.add_body(name="drone", mocap=True, pos=[40, 0, 1])
    frame_rgb, duct_rgb = [0.10, 0.10, 0.11, 1.0], [0.85, 0.20, 0.15, 1.0]
    drone.add_geom(name="drone_body", type=G.mjGEOM_BOX, size=[0.018, 0.013, 0.010], pos=[0, 0, 0.004],
                   rgba=frame_rgb, contype=0, conaffinity=0)
    drone.add_geom(name="drone_cam", type=G.mjGEOM_BOX, size=[0.005, 0.006, 0.005], pos=[0.020, 0, 0.010],
                   rgba=[0.05, 0.05, 0.05, 1.0], contype=0, conaffinity=0)
    for i, (dx, dy) in enumerate(WHOOP_MOTORS):
        cx, cy = WHOOP_ARM * dx, WHOOP_ARM * dy
        yaw = np.arctan2(cy, cx)
        drone.add_geom(name=f"drone_arm{i}", type=G.mjGEOM_BOX, size=[0.012, 0.003, 0.002],
                       pos=[cx / 2, cy / 2, 0], quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)],
                       rgba=frame_rgb, contype=0, conaffinity=0)
        for k in range(8):                       # duct wall: a ring of 8 thin panels
            a = 2 * np.pi * k / 8
            drone.add_geom(name=f"drone_duct{i}_{k}", type=G.mjGEOM_BOX, size=[0.0072, 0.0008, 0.006],
                           pos=[cx + WHOOP_DUCT_R * np.cos(a), cy + WHOOP_DUCT_R * np.sin(a), 0.002],
                           quat=[np.cos((a + np.pi / 2) / 2), 0, 0, np.sin((a + np.pi / 2) / 2)],
                           rgba=duct_rgb, contype=0, conaffinity=0)
    for i in range(4):
        rotor = w.add_body(name=f"drone_rotor{i}", mocap=True, pos=[40, 0, 1])
        rotor.add_geom(name=f"drone_rotor{i}", type=G.mjGEOM_BOX, size=[0.0155, 0.0028, 0.0005],
                       rgba=[0.9, 0.9, 0.92, 1.0], contype=0, conaffinity=0)

    # Event camera on the head: looks along +x.
    spec.body("base").add_camera(name="eye", pos=[0.33, 0.0, 0.05],
                                 xyaxes=[0, -1, 0, 0, 0, 1], fovy=60)
    third_pos = [-1.1, -1.9, 0.95]
    w.add_camera(name="third", pos=third_pos, xyaxes=_look_at(third_pos, [1.3, 0.1, 0.15]), fovy=50)
    return spec.compile()


class Go2Sim:
    """Standing Go2 with a PD pose controller and a ball launcher."""

    def __init__(self, seed: int = 0, eye_hw: tuple[int, int] = (180, 320),
                 third_hw: tuple[int, int] = (360, 640), scene: str = "lab", fast_eye: bool = False):
        """`fast_eye`: render the head camera without the robot's own meshes and without
        shadows. About 20x faster on integrated graphics (Intel Iris: 7 ms instead of 140 ms
        per image); the forward-looking camera hardly sees the robot anyway. Off by default so
        existing datasets stay reproducible."""
        self.rng = np.random.default_rng(seed)
        self.scene = scene
        self.model = build_model(self.rng, scene)
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
        self.damped = False      # True: motors only damp, as after Unitree's Damp command
        # Offscreen renderers are created on first use, so the interactive viewer
        # (which owns its own window and GL context) never has to create them.
        self._eye: mujoco.Renderer | None = None
        self._third: mujoco.Renderer | None = None
        self.eye_hw, self.third_hw = eye_hw, third_hw
        self._eye_cam = m.camera("eye").id
        self._eye_pos0 = m.cam_pos[self._eye_cam].copy()
        self._eye_quat0 = m.cam_quat[self._eye_cam].copy()
        self._eye_fovy0 = float(m.cam_fovy[self._eye_cam])
        self._mount_pos = self._eye_pos0.copy()     # per-episode camera mount
        self._mount_quat = self._eye_quat0.copy()
        self._shake: tuple | None = None             # (amplitudes, frequencies, phases)
        self._walker_mocap = m.body("walker").mocapid[0]
        self._walker_geoms = [m.geom(f"walker_{n}").id for n in ("torso", "leg_a", "leg_b")]
        self._drone_mocap = m.body("drone").mocapid[0]
        self._rotor_mocap = [m.body(f"drone_rotor{i}").mocapid[0] for i in range(4)]
        self.fast_eye = fast_eye
        self._eye_opt = mujoco.MjvOption()
        if fast_eye:
            for g in {int(m.geom_group[i]) for i in range(m.ngeom) if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH}:
                self._eye_opt.geomgroup[g] = 0
        # Optional walking (reflex/sim/locomotion.py). Off unless enable_walking() is called.
        self.gait = None
        self.mode = "pose"                  # "pose": fixed postures; "walk": the policy drives the legs
        self.command = np.zeros(3, dtype=np.float32)   # vx, vy (m/s), yaw rate (rad/s)
        self.slide_cmd = np.zeros(3)        # set_slide(): commanded vx, vy, wz without the policy
        self.slide_vel = np.zeros(3)        # and the speed it has reached
        self._passive = (m.dof_damping[self._v_adr].copy(), m.dof_frictionloss[self._v_adr].copy())
        self._contact = (int(m.opt.cone), float(m.opt.impratio))
        self._steps = 0
        self._motion: list | None = None      # keyframes left to play, see MOTIONS
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
        self.damped = False      # True: motors only damp, as after Unitree's Damp command
        self._set_mode("pose")
        self.command[:] = 0
        self.slide_cmd[:] = 0
        self.slide_vel[:] = 0
        self._motion = None
        self.park_ball()
        self.park_walker()
        self.park_drone()
        for _ in range(int(settle_s / self.dt)):
            self.step()
        d.time = 0.0

    def set_pose(self, name: str) -> None:
        self._set_mode("pose")
        self.command[:] = 0
        self.slide_cmd[:] = 0
        if name == "stand" and self.sitting:
            self.play("rise")           # getting up from sitting in one jump is violent
            return
        self._motion = None
        self.q_des = POSES[name].copy()

    def play(self, name: str) -> None:
        """Start a slow trick: sit, rise, hello. Joint targets move through its keyframes."""
        self._set_mode("pose")
        self.slide_cmd[:] = 0
        self.command[:] = 0
        self._motion = [[float(t), q.copy(), None] for t, q in MOTIONS[name]]

    @property
    def sitting(self) -> bool:
        return self._motion is None and bool(np.allclose(self.q_des, _SIT))

    @property
    def busy(self) -> bool:
        """True while a trick is still playing."""
        return self._motion is not None

    def _advance_motion(self) -> None:
        duration, target, start = self._motion[0]
        if start is None:                               # first step of this keyframe
            start = self._motion[0][2] = (self.data.time, self.q_des.copy())
        t0, q0 = start
        a = min(1.0, (self.data.time - t0) / max(duration, 1e-6))
        self.q_des = q0 + (target - q0) * a
        if a >= 1.0:
            self._motion.pop(0)
            if not self._motion:
                self._motion = None

    # --- optional walking ----------------------------------------------------------------
    def enable_walking(self, policy_path=None) -> None:
        """Load the walking policy. Nothing changes until set_velocity() is called."""
        from reflex.sim.locomotion import DEFAULT_POLICY, WalkPolicy
        self.gait = WalkPolicy(policy_path or DEFAULT_POLICY)
        self._policy_every = max(1, round(self.gait.step_dt / self.dt))

    def set_velocity(self, vx: float, vy: float, wz: float) -> None:
        """Walk: forward and sideways speed (m/s, +y is left) and turn rate (rad/s, + is left).

        Zero keeps the dog standing under the policy. Call set_pose() to go back to postures.
        """
        if self.gait is None:
            raise RuntimeError("call enable_walking() first")
        self.command[:] = self.gait.clip_command([vx, vy, wz])
        self._motion = None
        self._set_mode("walk")

    def set_slide(self, vx: float, vy: float, wz: float) -> None:
        """Sidestep, walk or turn without the walking policy: the dog keeps its posture and
        is moved as a whole with the speed profile of the real Go2 (SLIDE_TAU_*). Zero stops.

        A stand-in for sport mode's walking when `enable_walking()` is not used. The legs do
        not step; where the dog ends up and when is what matches the real robot.
        """
        if self.mode != "pose":
            raise RuntimeError("set_slide() is for postures; use set_velocity() while walking")
        self.slide_cmd[:] = (vx, vy, wz)

    def _advance_slide(self) -> None:
        if not (self.slide_cmd.any() or self.slide_vel.any()):
            return
        moving = self.slide_cmd != 0
        tau = np.where(moving, SLIDE_TAU_UP, SLIDE_TAU_DOWN)
        self.slide_vel += (self.slide_cmd - self.slide_vel) * min(1.0, self.dt) / tau
        self.slide_vel[np.abs(self.slide_vel) < 1e-4] = 0.0
        vx, vy, wz = self.slide_vel * self.dt
        q = self.data.qpos
        _, _, h = self.base_pose()
        q[0] += vx * np.cos(h) - vy * np.sin(h)
        q[1] += vx * np.sin(h) + vy * np.cos(h)
        if wz:
            turn = np.array([np.cos(wz / 2), 0.0, 0.0, np.sin(wz / 2)])
            out = np.zeros(4)
            mujoco.mju_mulQuat(out, turn, q[3:7])
            q[3:7] = out

    def _set_mode(self, mode: str) -> None:
        if mode == self.mode:
            return
        m = self.model
        if mode == "walk":
            # The policy was trained on joints without passive damping or friction loss.
            m.dof_damping[self._v_adr] = 0.0
            m.dof_frictionloss[self._v_adr] = 0.0
            # Contact model used in training. With the Menagerie one (elliptic cone, impratio
            # 100) the same policy walks at 50-70 % of the commanded speed instead of 70-85 %.
            m.opt.cone, m.opt.impratio = mujoco.mjtCone.mjCONE_PYRAMIDAL, 1.0
            self.gait.reset()
            self._steps = 0
        else:
            m.dof_damping[self._v_adr], m.dof_frictionloss[self._v_adr] = self._passive
            m.opt.cone, m.opt.impratio = self._contact
        self.mode = mode

    def walk_towards(self, x: float, y: float, heading: float = 0.0, tol: float = 0.06) -> bool:
        """Set the velocity command that takes the dog to a pose. True once it is there.

        Uses the simulator's ground truth; a real robot would need odometry for this.
        """
        px, py, h = self.base_pose()
        ex, ey = x - px, y - py
        eh = (heading - h + np.pi) % (2 * np.pi) - np.pi
        if np.hypot(ex, ey) < tol and abs(eh) < 0.06:
            if self.gait is None:
                self.set_slide(0.0, 0.0, 0.0)
            else:
                self.set_velocity(0.0, 0.0, 0.0)
            return True
        fwd, left = ex * np.cos(h) + ey * np.sin(h), -ex * np.sin(h) + ey * np.cos(h)
        cmd = (float(np.clip(2.5 * fwd, -0.5, 0.6)), float(np.clip(4.0 * left, -0.5, 0.5)),
               float(np.clip(2.5 * eh, -1.0, 1.0)))
        if self.gait is None:
            self.set_slide(*cmd)
        else:
            self.set_velocity(*cmd)
        return False

    def base_pose(self) -> tuple[float, float, float]:
        """x, y (m) and heading (rad) of the dog."""
        w, x, y, z = self.data.qpos[3:7]
        return float(self.data.qpos[0]), float(self.data.qpos[1]), float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def upright(self) -> float:
        """1.0 when the trunk is level, 0 on its side, negative upside down."""
        w, x, y, z = self.data.qpos[3:7]
        return float(1 - 2 * (x * x + y * y))

    def park_ball(self) -> None:
        a, v = self._ball_qadr, self._ball_vadr
        self.data.qpos[a:a + 7] = [20, 0, 0.2, 1, 0, 0, 0]
        self.data.qvel[v:v + 6] = 0

    def launch(self, shot: BallShot) -> None:
        """Roll a ball from its start point towards (0.35, y_target), just in front of the dog."""
        start = np.array([shot.x_start, shot.y_start])
        direction = np.array([0.35, shot.y_target]) - start
        direction /= np.linalg.norm(direction)
        self.launch_raw(start, shot.speed * direction, shot.radius, shot.rgb)

    def launch_raw(self, start_xy, vel_xy, radius: float, rgb) -> None:
        """Roll a ball from `start_xy` with velocity `vel_xy` (world frame, m and m/s)."""
        m, d = self.model, self.data
        m.geom_size[self._ball_geom, 0] = radius
        m.geom_rbound[self._ball_geom] = radius
        m.geom_rgba[self._ball_geom, :3] = rgb
        a, v = self._ball_qadr, self._ball_vadr
        d.qpos[a:a + 7] = [start_xy[0], start_xy[1], radius, 1, 0, 0, 0]
        # Pure rolling: omega = z x v / r
        d.qvel[v:v + 6] = [vel_xy[0], vel_xy[1], 0, -vel_xy[1] / radius, vel_xy[0] / radius, 0]

    def park_walker(self) -> None:
        self.data.mocap_pos[self._walker_mocap] = [30, 0, 0]

    def set_walker(self, x: float, y: float, bob: float = 0.0, rgb=None) -> None:
        """Place the walker (kinematic). Call every step to move it."""
        self.data.mocap_pos[self._walker_mocap] = [x, y, bob]
        if rgb is not None:
            self.model.geom_rgba[self._walker_geoms, :3] = rgb

    def park_drone(self) -> None:
        self.set_drone([40.0, 0.0, 1.0])

    def set_drone(self, pos, yaw: float = 0.0, rotor_rev_s: float = 250.0) -> None:
        """Place the drone (kinematic) and spin its props. Call every step to move it.

        Tiny-whoop props turn at roughly 20,000 rpm and more; rendered at a few hundred Hz
        that aliases, which still gives the flickering prop discs an event camera sees."""
        d = self.data
        pos = np.asarray(pos, float)
        c, s = np.cos(yaw), np.sin(yaw)
        d.mocap_pos[self._drone_mocap] = pos
        d.mocap_quat[self._drone_mocap] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        for i, (dx, dy) in enumerate(WHOOP_MOTORS):
            off = WHOOP_ARM * np.array([dx, dy])
            d.mocap_pos[self._rotor_mocap[i]] = pos + [c * off[0] - s * off[1], s * off[0] + c * off[1], 0.006]
            spin = yaw + (1 if i in (0, 3) else -1) * 2 * np.pi * rotor_rev_s * d.time
            d.mocap_quat[self._rotor_mocap[i]] = [np.cos(spin / 2), 0, 0, np.sin(spin / 2)]

    def set_ball_kinematic(self, xy, quat, radius: float | None = None, rgb=None) -> None:
        """Put the ball at `xy` with orientation `quat`, at rest on the floor.

        For scripted trajectories (e.g. a ball that rolls and comes to rest at a chosen
        point): set it before every step(); the physics does not move it in between.
        """
        m, d = self.model, self.data
        if radius is not None:
            m.geom_size[self._ball_geom, 0] = radius
            m.geom_rbound[self._ball_geom] = radius
        if rgb is not None:
            m.geom_rgba[self._ball_geom, :3] = rgb
        r = float(m.geom_size[self._ball_geom, 0])
        a, v = self._ball_qadr, self._ball_vadr
        d.qpos[a:a + 7] = [xy[0], xy[1], r, *quat]
        d.qvel[v:v + 6] = 0

    def set_camera_mount(self, offset, pitch_deg: float = 0.0, fovy: float | None = None) -> None:
        """Mount the head camera at `offset` (m, robot base frame) with a pitch (negative
        looks down) and a vertical field of view."""
        m, c = self.model, self._eye_cam
        pitch = np.radians(pitch_deg)
        self._mount_pos = np.asarray(offset, float)
        dq = np.array([np.cos(pitch / 2), np.sin(pitch / 2), 0, 0])
        self._mount_quat = np.zeros(4)
        mujoco.mju_mulQuat(self._mount_quat, self._eye_quat0, dq)
        m.cam_pos[c], m.cam_quat[c] = self._mount_pos, self._mount_quat
        if fovy is not None:
            m.cam_fovy[c] = fovy

    def set_head(self, yaw: float, pitch_down: float = 0.0) -> None:
        """Turn the camera like a body posture change while the legs stay planted.

        The Go2 has no head: in sport mode, looking somewhere means rotating the trunk with
        the Euler command (yaw, pitch). The camera on its mast then swings about the trunk
        center. Here only the camera mount is rotated about the base origin, by `yaw` (rad,
        + = left) and `pitch_down` (rad, + = nose down); the legs do not move. Relative to the
        mount set with set_camera_mount(). The real Euler limits are not known."""
        m, c = self.model, self._eye_cam
        cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch_down), np.sin(pitch_down)
        rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        rot = rz @ ry
        q_rot = np.zeros(4)
        mujoco.mju_mat2Quat(q_rot, rot.flatten())
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, q_rot, self._mount_quat)
        m.cam_pos[c] = rot @ self._mount_pos
        m.cam_quat[c] = q

    def project_to_eye(self, point) -> tuple[float, float, bool, float]:
        """Pixel (x, y) of a world point in the head camera image, whether it is inside the
        image and in front of the camera, and its distance (m). Uses the camera pose of the
        last step()."""
        m, d, c = self.model, self.data, self._eye_cam
        h, w = self.eye_hw
        rel = np.asarray(point, float) - d.cam_xpos[c]
        pc = d.cam_xmat[c].reshape(3, 3).T @ rel          # camera frame: x right, y up, looks along -z
        dist = float(np.linalg.norm(rel))
        if pc[2] >= -1e-6:
            return -1.0, -1.0, False, dist
        f = (h / 2) / np.tan(np.radians(m.cam_fovy[c]) / 2)
        px = w / 2 + f * pc[0] / -pc[2]
        py = h / 2 - f * pc[1] / -pc[2]
        return float(px), float(py), bool(0 <= px < w and 0 <= py < h), dist

    def randomize_camera(self, rng: np.random.Generator) -> dict:
        """Per-episode camera mount: we do not know where the real camera will sit."""
        m, c = self.model, self._eye_cam
        dz = rng.uniform(-0.05, 0.25)            # from head height up to on top of the back
        dy = rng.uniform(-0.04, 0.04)
        pitch = np.radians(rng.uniform(-12, 6))   # negative looks down
        fovy = rng.uniform(45, 75)
        self._mount_pos = self._eye_pos0 + [0, dy, dz]
        # Pitch is a rotation about the camera's own x axis (its "right").
        dq = np.array([np.cos(pitch / 2), np.sin(pitch / 2), 0, 0])
        self._mount_quat = np.zeros(4)
        mujoco.mju_mulQuat(self._mount_quat, self._eye_quat0, dq)
        m.cam_pos[c], m.cam_quat[c], m.cam_fovy[c] = self._mount_pos, self._mount_quat, fovy
        return {"cam_dz": float(dz), "cam_pitch_deg": float(np.degrees(pitch)), "cam_fovy": float(fovy)}

    def reset_camera(self) -> None:
        m, c = self.model, self._eye_cam
        self._mount_pos, self._mount_quat = self._eye_pos0.copy(), self._eye_quat0.copy()
        m.cam_pos[c], m.cam_quat[c], m.cam_fovy[c] = self._mount_pos, self._mount_quat, self._eye_fovy0
        self._shake = None

    def set_shake(self, rng: np.random.Generator, level: float) -> None:
        """Camera shake standing in for the balance motion of a real dog that "stands still".

        `level` 0 disables it; 1 is about 0.4 degrees and 2 mm, a guess until we measure
        the real robot on Saturday.
        """
        if level <= 0:
            self._shake = None
            return
        amp = level * np.array([np.radians(0.4), np.radians(0.4), 0.002]) * rng.uniform(0.3, 1.0, 3)
        self._shake = (amp, rng.uniform(1.5, 9.0, 3), rng.uniform(0, 2 * np.pi, 3))

    def _apply_shake(self) -> None:
        if self._shake is None:
            return
        amp, freq, phase = self._shake
        pitch, roll, dz = amp * np.sin(2 * np.pi * freq * self.data.time + phase)
        dq = np.array([1.0, pitch / 2, 0.0, roll / 2])   # small-angle quaternion
        dq /= np.linalg.norm(dq)
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, self._mount_quat, dq)
        c = self._eye_cam
        self.model.cam_quat[c] = q
        self.model.cam_pos[c] = self._mount_pos + [0, 0, dz]

    def ball_xy(self) -> np.ndarray:
        return self.data.qpos[self._ball_qadr:self._ball_qadr + 2].copy()

    def step(self) -> None:
        d = self.data
        if self.mode == "walk" and not self.damped:
            g = self.gait
            if self._steps % self._policy_every == 0:
                g.act(d.qvel[3:6], d.qpos[3:7], self.command, d.qpos[self._q_adr], d.qvel[self._v_adr])
            self._steps += 1
            tau = g.kp * (g.target - d.qpos[self._q_adr]) - g.kd * d.qvel[self._v_adr]
            d.ctrl[:] = np.clip(tau, self._ctrl_lo, self._ctrl_hi)
            mujoco.mj_step(self.model, d)
            return
        if self._motion is not None:
            self._advance_motion()
        kp = 0.0 if self.damped else KP
        tau = kp * (self.q_des - d.qpos[self._q_adr]) - KD * d.qvel[self._v_adr]
        d.ctrl[:] = np.clip(tau, self._ctrl_lo, self._ctrl_hi)
        mujoco.mj_step(self.model, d)
        self._advance_slide()

    def render_eye(self) -> np.ndarray:
        """RGB uint8 (H, W, 3) from the head camera."""
        if self._eye is None:
            self._eye = mujoco.Renderer(self.model, *self.eye_hw)
        self._apply_shake()
        self._eye.update_scene(self.data, camera="eye", scene_option=self._eye_opt)
        if self.fast_eye:
            self._eye.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            self._eye.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        return self._eye.render()

    def render_third(self) -> np.ndarray:
        if self._third is None:
            self._third = mujoco.Renderer(self.model, *self.third_hw)
        self._third.update_scene(self.data, camera="third")
        return self._third.render()


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 -> luminance float32 in [0, 1]."""
    return (rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)) / 255.0
