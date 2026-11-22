"""The whole pipeline on the Kria KV260: camera -> FPGA SNN -> decision -> sport-mode request.

    ros2 launch reflex_ros kria.launch.py robot:=sim              # simulator on the laptop is the robot
    ros2 launch reflex_ros kria.launch.py robot:=real enabled:=true   # the real Go2 (after checks)

Runs on the board (scripts/kria/run.sh sources everything and checks the camera first):

  despot_send_and_forget   GenX320 -> 2x80x80 frames -> port 3333, `despot:=false` to skip
  camera_tap               3333 -> spleat-server on 3334, unchanged, plus /reflex/debug/events_image
                           (what the network sees, for ros-web); `tap:=false`: despot -> server directly
  fpga_perception          spleat-server's output -> /reflex/decision (readout of `model_dir`:
                           models/zones3 or models/ball_v3, which must match what is on the FPGA)
  behavior                 /reflex/decision -> /api/sport/request (all safety logic); with
                           mode:=look (models/ball_v3/run_args) /durance/ball_zone instead: the
                           trunk points at the ball's zone, six zones -> yaw and pitch

Only the robot changes. `robot` picks the network the DDS traffic leaves on:
  real  /etc/cyclonedds/cyclonedds-go2.xml (USB-Ethernet adapter, 192.168.123.99) -> the Go2
  sim   the board's eth0 (192.168.77.1) -> the laptop, where `ros-robot-sim behavior:=false`
        runs the MuJoCo dog as the robot only

Safety: with robot:=real the behavior node starts disabled (it logs what it would send);
pass enabled:=true on purpose. min_confidence defaults to 0.45: zone decisions from the
FPGA reached 0.5-0.55 with balls thrown past the camera on 22 Nov (not validated).
"""
import json
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction, SetEnvironmentVariable,
                            TimerAction)
from launch.substitutions import LaunchConfiguration as LC
from launch_ros.actions import Node

ARGS = {
    "robot": "sim",                 # real | sim
    "enabled": "auto",              # auto: true for sim, false for real
    "despot": "true",               # start the camera client here
    "tap": "true",                  # camera_tap between despot and the server (server on 3334)
    "model_dir": "/opt/reflex/models/zones3",
    "host": "127.0.0.1", "port": "auto", "margin": "0.0",   # port auto: 3334 with tap, else 3333
    "readout": "0",                 # frames per decision, 0: from model_dir
    "vote_window": "5", "vote_min": "3",   # 6-zone networks: a zone needs vote_min of the last vote_window decisions
    "zone_hold_s": "1.0",           # 6-zone networks: seconds a zone stays published before none
    "max_inflight": "4",            # camera_tap: frames on the FPGA at once; more are dropped (4: all of 50/s)
    "zone_rows": "auto",            # rows of the zone grid in the tap's picture; auto: from model_dir
    "mode": "keeper", "reaction": "euler", "min_confidence": "0.45", "consecutive": "2",
    "move_vy": "0.3", "hold_s": "0.6",
    # mode:=look: trunk gestures towards the six zones. Small angles until checked on the real Go2.
    "look_yaw": "0.15", "look_pitch": "0.1", "camera_on_trunk": "false", "invert_pitch": "false",
    "lost_s": "2.0",
}

DDS = {
    "real": "file:///etc/cyclonedds/cyclonedds-go2.xml",
    "sim": "<CycloneDDS><Domain Id=\"any\"><General><Interfaces>"
           "<NetworkInterface name=\"eth0\"/></Interfaces></General></Domain></CycloneDDS>",
}


def _bool(text: str) -> bool:
    return text.lower() in ("1", "true", "yes")


def _zone_rows(model_dir: str) -> int:
    """Rows of zones: `zones.rows` of a *_readout.json (models/ball_v3: 2), else 1 (zones3)."""
    found = sorted(Path(model_dir).glob("*_readout.json"))
    if len(found) == 1:
        return int(json.loads(found[0].read_text()).get("zones", {}).get("rows", 1))
    return 1


def _setup(context):
    robot = LC("robot").perform(context)
    if robot not in DDS:
        raise RuntimeError(f"robot:={robot}: expected real or sim")
    enabled = LC("enabled").perform(context)
    enabled = (robot == "sim") if enabled == "auto" else enabled.lower() in ("1", "true", "yes")
    actions = [
        SetEnvironmentVariable("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp"),
        SetEnvironmentVariable("CYCLONEDDS_URI", DDS[robot]),
        LogInfo(msg=f"robot {robot}, behavior enabled {enabled}"),
    ]
    tap = LC("tap").perform(context).lower() in ("1", "true", "yes")
    port = LC("port").perform(context)
    port = (3334 if tap else 3333) if port == "auto" else int(port)
    if tap:
        rows = LC("zone_rows").perform(context)
        rows = _zone_rows(LC("model_dir").perform(context)) if rows == "auto" else int(rows)
        actions.append(Node(package="reflex_ros", executable="camera_tap", output="screen",
                            parameters=[{"listen_port": 3333, "server_port": port, "zone_rows": rows,
                                         "max_inflight": int(LC("max_inflight").perform(context))}]))
    if LC("despot").perform(context).lower() in ("1", "true", "yes"):
        despot = ExecuteProcess(cmd=["despot_send_and_forget", "--to_display", "0"], output="screen", name="despot")
        actions.append(TimerAction(period=3.0, actions=[despot]) if tap else despot)   # tap listens first
    actions += [
        Node(package="reflex_ros", executable="fpga_perception", output="screen", parameters=[{
            "host": LC("host"), "port": port,
            "model_dir": LC("model_dir"), "margin": float(LC("margin").perform(context)),
            "readout": int(LC("readout").perform(context)),
            "vote_window": int(LC("vote_window").perform(context)),
            "vote_min": int(LC("vote_min").perform(context)),
            "zone_hold_s": float(LC("zone_hold_s").perform(context))}]),
        Node(package="reflex_ros", executable="behavior", output="screen", parameters=[{
            "mode": LC("mode"), "reaction": LC("reaction"), "enabled": enabled,
            "min_confidence": float(LC("min_confidence").perform(context)),
            "consecutive": int(LC("consecutive").perform(context)),
            "move_vy": float(LC("move_vy").perform(context)),
            "hold_s": float(LC("hold_s").perform(context)),
            "look_yaw": float(LC("look_yaw").perform(context)),
            "look_pitch": float(LC("look_pitch").perform(context)),
            "camera_on_trunk": _bool(LC("camera_on_trunk").perform(context)),
            "invert_pitch": _bool(LC("invert_pitch").perform(context)),
            "lost_s": float(LC("lost_s").perform(context))}]),
    ]
    return actions


def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument(k, default_value=v) for k, v in ARGS.items()]
                             + [OpaqueFunction(function=_setup)])
