from reflex.runtime import sport_api as api
from reflex.runtime.sim_sport import pose_for_request


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
