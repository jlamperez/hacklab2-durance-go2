"""camera_tap's in-place CHW -> HWC reorder gives the same message as decode + re-encode."""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from reflex.runtime.spleat_client import cbor_decode, cbor_encode

NODE = Path(__file__).resolve().parent.parent / "ros2_ws/src/reflex_ros/reflex_ros/camera_tap_node.py"


@pytest.fixture(scope="module")
def tap():
    """The node module with rclpy and sensor_msgs stubbed: only the pure functions are used."""
    stubs = {"rclpy": types.ModuleType("rclpy"), "rclpy.node": types.ModuleType("rclpy.node"),
             "sensor_msgs": types.ModuleType("sensor_msgs"), "sensor_msgs.msg": types.ModuleType("sensor_msgs.msg")}
    stubs["rclpy.node"].Node = object
    stubs["sensor_msgs.msg"].Image = object
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location("camera_tap_node", NODE)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k)
            else:
                sys.modules[k] = v
    node = mod.CameraTapNode.__new__(mod.CameraTapNode)
    node.to_hwc = True
    return node


@pytest.mark.parametrize("side", [80, 320])        # 2-byte and 4-byte CBOR lengths
def test_reorder_matches_decode_and_reencode(tap, side):
    frame = np.random.default_rng(0).random((2, side, side)).astype("<f4")
    body = cbor_encode({"id": 7, "payload": frame.tobytes(), "t": 1.5})
    got_frame, got = tap.reorder(body)
    want = cbor_encode({"id": 7, "payload": np.ascontiguousarray(frame.transpose(1, 2, 0)).tobytes(), "t": 1.5})
    assert got == want
    assert np.array_equal(got_frame, frame)
    assert cbor_decode(got)["id"] == 7


def test_non_frames_pass_unchanged(tap):
    body = cbor_encode({"id": 1, "note": "hello"})
    assert tap.reorder(body) == (None, body)
