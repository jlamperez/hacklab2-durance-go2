from reflex.sim.scene import POSES
from reflex.sim.view import KEY_POSES, pose_for_key


def test_keys_map_to_existing_poses():
    assert set(KEY_POSES.values()) == set(POSES)


def test_pose_keys():
    assert pose_for_key(ord("S")) == "stand"
    assert pose_for_key(ord("C")) == "crouch"
    assert pose_for_key(ord("L")) == "lean_left"
    assert pose_for_key(ord("R")) == "lean_right"


def test_other_keys_are_ignored():
    assert pose_for_key(ord("X")) is None
    assert pose_for_key(32) is None        # space: the viewer's own pause key
    assert pose_for_key(262) is None       # arrow key, outside the ASCII range
