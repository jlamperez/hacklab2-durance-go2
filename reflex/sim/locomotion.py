"""Optional walking for the simulated Go2: a learned policy turns a velocity command
(forward, sideways, turn) into joint targets, 50 times per second.

The policy was trained with reinforcement learning in unitree_rl_mjlab (see
`models/go2_walk.json` and `docs/go2-control.md`). Everything here mirrors how it was
trained: observation layout, default pose, action scale, joint gains.

This is low-level control. On the real robot, sport mode does this job; here it lets
the simulated dog answer a `Move` request by actually walking.
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

DEFAULT_POLICY = Path(__file__).resolve().parents[2] / "models" / "go2_walk.onnx"


class WalkPolicy:
    def __init__(self, path: str | Path = DEFAULT_POLICY):
        import onnxruntime as ort
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        self.step_dt = float(meta["step_dt"])
        self.default = np.array(meta["default_joint_pos"], dtype=np.float32)
        self.scale = float(meta["action_scale"])
        self.kp = np.tile(np.array(meta["stiffness"], dtype=np.float64), 4)
        self.kd = np.tile(np.array(meta["damping"], dtype=np.float64), 4)
        self.period = 0.6
        r = meta["command_ranges"]
        self.cmd_lo = np.array([r["vx"][0], r["vy"][0], r["wz"][0]], dtype=np.float32)
        self.cmd_hi = np.array([r["vx"][1], r["vy"][1], r["wz"][1]], dtype=np.float32)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self) -> None:
        self.last_action = np.zeros(12, dtype=np.float32)
        self.ticks = 0
        self.target = self.default.astype(np.float64)

    def clip_command(self, cmd) -> np.ndarray:
        return np.clip(np.asarray(cmd, dtype=np.float32), self.cmd_lo, self.cmd_hi)

    def observation(self, ang_vel_body, quat_wxyz, command, q, qd) -> np.ndarray:
        gravity = np.zeros(3)
        inv = np.zeros(4)
        mujoco.mju_negQuat(inv, np.asarray(quat_wxyz, dtype=np.float64))
        mujoco.mju_rotVecQuat(gravity, np.array([0.0, 0.0, -1.0]), inv)
        if np.linalg.norm(command) < 0.1:
            phase = np.zeros(2)
        else:
            g = (self.ticks * self.step_dt) % self.period / self.period
            phase = np.array([np.sin(2 * np.pi * g), np.cos(2 * np.pi * g)])
        return np.concatenate([ang_vel_body, gravity, command, phase, q - self.default, qd,
                               self.last_action]).astype(np.float32)

    def act(self, ang_vel_body, quat_wxyz, command, q, qd) -> np.ndarray:
        """One policy step. Returns the 12 joint position targets."""
        obs = self.observation(ang_vel_body, quat_wxyz, command, q, qd)
        action = self.session.run(None, {"obs": obs[None]})[0][0]
        self.last_action = action.astype(np.float32)
        self.ticks += 1
        self.target = (self.default + self.scale * action).astype(np.float64)
        return self.target
