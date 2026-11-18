import numpy as np
import pytest

from reflex.sim.scene import GO2_XML, BallShot, Go2Sim

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
