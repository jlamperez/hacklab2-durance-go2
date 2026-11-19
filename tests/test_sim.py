import mujoco
import numpy as np
import pytest

from reflex.sim.scene import GO2_XML, POSES, BallShot, Go2Sim

pytestmark = pytest.mark.skipif(not GO2_XML.exists(), reason="Go2 model missing: run `pixi run setup`")


@pytest.fixture(scope="module")
def sim():
    return Go2Sim(seed=0)


def test_dog_stands_still_after_reset(sim):
    sim.reset()
    z0 = sim.data.qpos[2]
    for _ in range(250):
        sim.step()
    assert 0.25 < sim.data.qpos[2] < 0.40
    assert abs(sim.data.qpos[2] - z0) < 0.01
    assert np.abs(sim.data.qpos[:2]).max() < 0.05


def test_crouch_lowers_the_trunk(sim):
    sim.reset()
    z_stand = sim.data.qpos[2]
    sim.set_pose("crouch")
    for _ in range(400):
        sim.step()
    assert sim.data.qpos[2] < z_stand - 0.08


@pytest.mark.parametrize("label,sign", [("left", 1), ("right", -1)])
def test_ball_arrives_on_the_labelled_side(sim, label, sign):
    rng = np.random.default_rng(1)
    shot = BallShot.sample(rng, label=label)
    sim.reset()
    sim.launch(shot)
    while sim.ball_xy()[0] > 0.6:
        sim.step()
    assert np.sign(sim.ball_xy()[1]) == sign   # +y is the dog's left


def test_eye_camera_renders(sim):
    img = sim.render_eye()
    assert img.shape == (*sim.eye_hw, 3) and img.std() > 5


def _hold(sim, pose, seconds=1.5):
    """Switch to `pose` from standing and return (y, z, roll_deg, max_speed) once settled."""
    sim.reset()
    sim.set_pose(pose)
    for _ in range(int(seconds / sim.dt)):
        sim.step()
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, sim.data.qpos[3:7])
    rot = rot.reshape(3, 3)
    roll = np.degrees(np.arctan2(rot[2, 1], rot[2, 2]))
    return sim.data.qpos[1], sim.data.qpos[2], roll, np.abs(sim.data.qvel[:6]).max()


@pytest.mark.parametrize("pose", list(POSES))
def test_every_pose_is_stable(sim, pose):
    """The dog settles (does not fall, drift or keep moving) in every pose."""
    y, z, roll, speed = _hold(sim, pose)
    assert speed < 0.02, "still moving after 1.5 s"
    assert z > 0.10, "trunk on the floor"
    assert abs(roll) < 25, "rolled over"
    assert abs(y) < 0.15, "drifted sideways"


@pytest.mark.parametrize("pose,sign", [("lean_left", 1), ("lean_right", -1)])
def test_lean_moves_the_trunk_to_the_named_side(sim, pose, sign):
    y, z, roll, _ = _hold(sim, pose)
    assert sign * y > 0.02                 # +y is the dog's left
    assert sign * roll > 5                 # and it rolls towards that side
    assert z > 0.25                        # while staying up


def test_leans_are_mirror_images(sim):
    y_l, z_l, roll_l, _ = _hold(sim, "lean_left")
    y_r, z_r, roll_r, _ = _hold(sim, "lean_right")
    assert abs(y_l + y_r) < 0.005 and abs(roll_l + roll_r) < 1.0 and abs(z_l - z_r) < 0.005


def test_dog_recovers_to_stand_from_every_pose(sim):
    sim.reset()
    z_stand = sim.data.qpos[2]
    for pose in ("crouch", "lean_left", "lean_right"):
        sim.set_pose(pose)
        for _ in range(500):
            sim.step()
        sim.set_pose("stand")
        for _ in range(750):
            sim.step()
        assert abs(sim.data.qpos[2] - z_stand) < 0.02, f"did not stand back up after {pose}"
        assert abs(sim.data.qpos[1]) < 0.05


# --- optional walking --------------------------------------------------------------------
def _walk(sim, cmd, seconds):
    x0, y0, h0 = sim.base_pose()
    sim.set_velocity(*cmd)
    upright = 1.0
    for _ in range(int(seconds / sim.dt)):
        sim.step()
        upright = min(upright, sim.upright())
    x, y, h = sim.base_pose()
    dx, dy = x - x0, y - y0
    return dx * np.cos(h0) + dy * np.sin(h0), -dx * np.sin(h0) + dy * np.cos(h0), h - h0, upright


def test_walking_policy_follows_commands_and_stays_up():
    sim = Go2Sim(seed=0)
    sim.enable_walking()
    fwd, left, turn, up = _walk(sim, (0.8, 0, 0), 3.0)
    assert fwd > 1.0 and abs(left) < 0.3 and up > 0.95
    fwd, left, turn, up = _walk(sim, (0, 0.4, 0), 3.0)
    assert left > 0.25 and up > 0.95
    fwd, left, turn, up = _walk(sim, (0, 0, 0.8), 3.0)
    assert turn > 0.8 and up > 0.95
    fwd, left, turn, up = _walk(sim, (0, 0, 0), 2.0)
    assert abs(fwd) < 0.15 and abs(left) < 0.1 and up > 0.95


def test_postures_still_work_after_walking():
    sim = Go2Sim(seed=0)
    damping = sim.model.dof_damping.copy()
    sim.enable_walking()
    _walk(sim, (0.5, 0, 0), 1.0)
    sim.set_pose("crouch")
    assert sim.mode == "pose" and np.array_equal(sim.model.dof_damping, damping)
    for _ in range(600):
        sim.step()
    assert sim.data.qpos[2] < 0.2 and sim.upright() > 0.95


def test_walking_is_off_by_default():
    sim = Go2Sim(seed=0)
    assert sim.gait is None and sim.mode == "pose"
    with pytest.raises(RuntimeError):
        sim.set_velocity(0.5, 0, 0)
