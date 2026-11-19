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


def test_drive_keys_add_speed_and_space_stops():
    from reflex.sim.view import KEY_DOWN, KEY_LEFT, KEY_SPACE, KEY_UP, drive_command
    cmd = drive_command(KEY_UP, (0, 0, 0))
    assert cmd == (0.2, 0.0, 0.0)
    assert drive_command(KEY_UP, cmd) == (0.4, 0.0, 0.0)
    assert drive_command(KEY_DOWN, cmd) == (0.0, 0.0, 0.0)
    assert drive_command(KEY_LEFT, cmd) == (0.2, 0.0, 0.3)
    assert drive_command(ord("A"), (0, 0, 0)) == (0.0, 0.15, 0.0)
    assert drive_command(KEY_SPACE, (0.6, 0.1, 0.3)) == (0.0, 0.0, 0.0)
    assert drive_command(ord("Z"), (0, 0, 0)) is None


def test_drive_keys_respect_limits():
    from reflex.sim.view import KEY_UP, drive_command
    cmd = (0, 0, 0)
    for _ in range(20):
        cmd = drive_command(KEY_UP, cmd)
    assert cmd == (1.0, 0.0, 0.0)
