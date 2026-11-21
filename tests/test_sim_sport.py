from reflex.runtime import sport_api as api
from reflex.runtime.sim_sport import euler_for_request, pose_for_request


def test_move_and_euler_lean_to_the_commanded_side():
    assert pose_for_request(api.MOVE.api_id, api.move_params(0, 0.3, 0)) == "lean_left"
    assert pose_for_request(api.MOVE.api_id, api.move_params(0, -0.3, 0)) == "lean_right"
    # Right-hand rule about +x: positive roll lowers the right side (checked on the real Go2).
    assert pose_for_request(api.EULER.api_id, api.euler_params(0.3, 0, 0)) == "lean_right"
    assert pose_for_request(api.EULER.api_id, api.euler_params(-0.3, 0, 0)) == "lean_left"


def test_zero_commands_and_stop_return_to_stand():
    assert pose_for_request(api.MOVE.api_id, api.move_params(0, 0, 0)) == "stand"
    assert pose_for_request(api.EULER.api_id, api.euler_params(0, 0, 0)) == "stand"
    for cmd in (api.STOP_MOVE, api.STAND_UP, api.BALANCE_STAND, api.RECOVERY_STAND):
        assert pose_for_request(cmd.api_id) == "stand"


def test_stand_down_and_damp():
    assert pose_for_request(api.STAND_DOWN.api_id) == "crouch"
    assert pose_for_request(api.DAMP.api_id) == "damp"


def test_unknown_or_malformed_requests_are_ignored():
    assert pose_for_request(9999) is None
    assert pose_for_request(api.DANCE1.api_id) is None
    assert pose_for_request(api.FRONT_FLIP.api_id) is None
    assert pose_for_request(api.MOVE.api_id, "not json") is None


def test_sit_and_hello_have_a_simulated_version():
    assert pose_for_request(api.SIT.api_id) == "sit"
    assert pose_for_request(api.RISE_SIT.api_id) == "stand"
    assert pose_for_request(api.HELLO.api_id) == "hello"


def test_posture_mode_gates_euler_and_move_like_the_real_go2():
    from reflex.runtime.sim_sport import PostureMode
    pm = PostureMode()
    lean = api.euler_params(0.3, 0, 0)
    step = api.move_params(0, 0.3, 0)
    assert not pm.handle(api.EULER.api_id, lean)          # outside posture mode: no effect
    assert pm.handle(api.MOVE.api_id, step)               # but Move walks
    assert not pm.handle(api.POSE.api_id, api.pose_params(True))
    assert pm.handle(api.EULER.api_id, lean)              # inside: Euler leans
    assert not pm.handle(api.MOVE.api_id, step)           # and Move does not walk
    pm.handle(api.POSE.api_id, api.pose_params(False))
    assert pm.handle(api.MOVE.api_id, step)
    assert pm.handle(api.BALANCE_STAND.api_id)


def test_behavior_sequences_take_effect_in_posture_mode():
    """The requests behavior sends for a lean and back all do what they should."""
    from reflex.runtime.sim_sport import PostureMode
    pm = PostureMode()
    there = api.requests_for_pose("lean_left", "stand", "euler", 0.3, 0.3)
    back = api.requests_for_pose("stand", "lean_left", "euler", 0.3, 0.3)
    effects = [(c.name, pm.handle(c.api_id, p)) for c, p in there + back]
    assert effects == [("Pose", False), ("Euler", True), ("Euler", True), ("Pose", False), ("BalanceStand", True)]
    assert pm.handle(api.MOVE.api_id, api.move_params(0, 0.3, 0))   # free to walk afterwards


def test_euler_with_yaw_or_pitch_turns_the_trunk_and_roll_alone_still_leans():
    assert euler_for_request(api.EULER.api_id, api.euler_params(0.0, 0.2, -0.3)) == (0.0, 0.2, -0.3)
    assert euler_for_request(api.EULER.api_id, api.euler_params(0.3, 0.0, 0.0)) is None
    assert euler_for_request(api.EULER.api_id, api.euler_params(0.0, 0.0, 0.0)) is None
    assert euler_for_request(api.MOVE.api_id, api.move_params(0.0, 0.3, 0.0)) is None
    assert euler_for_request(api.EULER.api_id, "not json") is None
