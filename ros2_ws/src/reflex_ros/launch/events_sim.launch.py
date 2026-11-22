"""Real events, simulated robot: a recording (or the live KV260 camera) drives the MuJoCo dog.

    ros2 launch reflex_ros events_sim.launch.py source:=data/river/sat_follow2
    ros2 launch reflex_ros events_sim.launch.py source:=data/river/sat_follow2/ball/take_007_center.h5
    ros2 launch reflex_ros events_sim.launch.py source:=live kria_host:=root@durance-kria.local
    ros2 launch reflex_ros events_sim.launch.py source:=... behavior:=reflex
    ros2 launch reflex_ros events_sim.launch.py source:=data/river/sat_follow2 tracker:=snn
    ros2 launch reflex_ros events_sim.launch.py tracker:=fpga mode:=look
    ros2 launch reflex_ros events_sim.launch.py source:=data/river/sat_follow2 tracker:=fpga-sim mode:=look

  event_camera (replay or live, 320x320) -> /reflex/events_window
  behavior:=follow   target node -> /reflex/target -> follow node -> Move / StopMove -> walking dog
                     mode:=look  nine trunk gestures (Euler yaw + pitch) instead of walking
                     tracker:=centroid  the target node, no learning (default)
                     tracker:=snn       ball_target: Theodor's SNN on /reflex/events_fullres, by default the
                                        zone classifier (ball_model:=out/models/ball_zones_best.pt: six zones
                                        or none; the target is the zone's centre). Needs PyTorch, riverops and
                                        Training/ from feat/training, see reflex/runtime/ball_snn.py
                     tracker:=fpga      the follow node reads the zone the Kria's ball_node publishes
                                        on /durance/ball_zone (input:=zone; the zones network on the
                                        FPGA). No event_camera here: the board reads the camera itself.
                                        Needs the Kria on the same ROS network: not from Docker on a Mac
                     tracker:=fpga-sim  as fpga, but ball_zone_sim stands in for the Kria's ball_node: the
                                        zones network on the CPU over the recording, majority-voted
                                        (ball_vote_window, ball_vote_min), on /durance/ball_zone
  behavior:=reflex   perception (simple detector) -> /reflex/decision -> behavior node -> posture

Open loop for recordings: the dog's motion does not change what the camera saw, so this
shows the commands real events would cause, not whether the dog would reach the ball.
The follow node starts enabled here (follow:=false to leave it out). Like ros-robot-sim it
publishes on /api/sport/request: keep the real Go2 off this ROS network (or use another
ROS_DOMAIN_ID). Run from the repository root so relative paths resolve.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ARGS = {
    # camera
    "source": "data/river", "speed": "1.0", "loop": "true", "gap_s": "1.0", "lead_s": "0.0", "still_first": "true",
    "exclude": "", "flip_x": "false", "flip_y": "false", "port": "3333", "kria_host": "",
    # what reacts
    "behavior": "follow",
    # follow
    "follow": "true", "mode": "walk", "hfov_deg": "90.0", "zone_top": "0.3", "min_events": "0.0",
    # mode:=look. Open loop: the recorded camera did not turn with the simulated trunk, so the
    # gesture is not added to the target's angle. live with the camera on the real dog: true.
    "vfov_deg": "90.0", "camera_on_trunk": "false",
    # follow: where the target comes from. snn: ball_model, ball_min_p (P(ball), or P(zone) for the
    # zones network), ball_min_pixels, fullres_ms (30 ms: zones on sat_follow2, docs/events-sim.md)
    "tracker": "centroid", "ball_model": "out/models/ball_zones_best.pt", "ball_min_p": "0.55",
    "ball_min_pixels": "20", "fullres_ms": "30.0",
    # fpga-sim: the stand-in ball_node's vote, a zone needs ball_vote_min of the last ball_vote_window
    "ball_vote_window": "5", "ball_vote_min": "3",
    # reflex
    "detector": "simple", "reaction": "euler", "min_confidence": "0.6", "consecutive": "2",
    "move_vy": "0.3", "hold_s": "0.6",
    # simulated robot. actuation_delay_ms: empty = 300 ms walking (as ros-follow-sim), 70 ms postures
    "actuation_delay_ms": "", "viewer": "true", "scene": "lab",
}


def _bool(s: str) -> bool:
    return s.strip().lower() in ("1", "true", "yes", "on")


def setup(context):
    a = {k: LaunchConfiguration(k).perform(context) for k in ARGS}
    behavior = a["behavior"]
    if behavior not in ("follow", "reflex"):
        raise ValueError(f"behavior:={behavior}: use follow or reflex")
    if a["tracker"] not in ("centroid", "snn", "fpga", "fpga-sim"):
        raise ValueError(f"tracker:={a['tracker']}: use centroid, snn, fpga or fpga-sim")
    if a["tracker"] in ("fpga", "fpga-sim") and behavior != "follow":
        raise ValueError(f"tracker:={a['tracker']} needs behavior:=follow")
    if behavior == "reflex" and a["detector"] != "simple":
        raise ValueError("behavior:=reflex works with detector:=simple only: the reflex SNN was trained "
                         "on 320x180 windows, this camera gives 320x320")
    walking = behavior == "follow"
    delay = float(a["actuation_delay_ms"]) if a["actuation_delay_ms"] else (300.0 if walking else 70.0)

    camera = Node(package="reflex_ros", executable="event_camera", output="screen", parameters=[{
        "source": a["source"], "speed": float(a["speed"]), "loop": _bool(a["loop"]),
        "gap_s": float(a["gap_s"]), "lead_s": float(a["lead_s"]), "still_first": _bool(a["still_first"]),
        "exclude": a["exclude"],
        "flip_x": _bool(a["flip_x"]), "flip_y": _bool(a["flip_y"]), "port": int(a["port"]),
        "kria_host": a["kria_host"], "fullres_ms": float(a["fullres_ms"])}])
    bridge = Node(package="reflex_ros", executable="sim_bridge", output="screen", parameters=[{
        "robot_only": True, "camera": False, "locomotion": "rl" if walking else "none",
        "viewer": _bool(a["viewer"]), "scene": a["scene"], "actuation_delay_ms": delay}],
        additional_env={"MUJOCO_GL": "glfw"})
    nodes = [bridge, RegisterEventHandler(OnProcessExit(target_action=bridge, on_exit=[Shutdown()]))]
    if a["tracker"] != "fpga":
        nodes.insert(0, camera)
    if walking:
        if a["tracker"] == "fpga":
            pass                          # the follow node subscribes to /durance/ball_zone itself
        elif a["tracker"] == "fpga-sim":
            nodes.append(Node(package="reflex_ros", executable="ball_zone_sim", output="screen", parameters=[{
                "source": "snn", "checkpoint": a["ball_model"], "min_p": float(a["ball_min_p"]),
                "min_pixels": int(a["ball_min_pixels"]), "vote_window": int(a["ball_vote_window"]),
                "vote_min": int(a["ball_vote_min"])}]))
        elif a["tracker"] == "snn":
            nodes.append(Node(package="reflex_ros", executable="ball_target", output="screen", parameters=[{
                "checkpoint": a["ball_model"], "min_p": float(a["ball_min_p"]),
                "min_pixels": int(a["ball_min_pixels"])}]))
        else:
            nodes.append(Node(package="reflex_ros", executable="target", output="screen", parameters=[{
                "zone_top": float(a["zone_top"]), "min_events": float(a["min_events"])}]))
        if _bool(a["follow"]):
            nodes.append(Node(package="reflex_ros", executable="follow", output="screen", parameters=[{
                "mode": a["mode"], "hfov_deg": float(a["hfov_deg"]), "vfov_deg": float(a["vfov_deg"]),
                "camera_on_trunk": _bool(a["camera_on_trunk"]), "autostart": True,
                "input": "zone" if a["tracker"] in ("fpga", "fpga-sim") else "target"}]))
    else:
        nodes.append(Node(package="reflex_ros", executable="perception", output="screen", parameters=[{
            "detector": "simple"}]))
        nodes.append(Node(package="reflex_ros", executable="behavior", output="screen", parameters=[{
            "mode": "keeper", "reaction": a["reaction"], "enabled": True,
            "min_confidence": float(a["min_confidence"]), "consecutive": int(a["consecutive"]),
            "move_vy": float(a["move_vy"]), "hold_s": float(a["hold_s"])}]))
    return nodes


def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument(k, default_value=v) for k, v in ARGS.items()]
                             + [OpaqueFunction(function=setup)])
