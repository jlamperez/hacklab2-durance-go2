"""Closed loop in simulation: MuJoCo bridge + perception + behavior.

    ros2 launch reflex_ros sim.launch.py detector:=snn max_shots:=20
    ros2 launch reflex_ros sim.launch.py detector:=simple actuation_delay_ms:=200

Run from the repository root so the relative model path resolves.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration as LC
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue as PV

ARGS = {
    "detector": "snn", "model": "out/models/snn.pt", "device": "cpu", "mode": "keeper",
    "reaction": "euler", "actuation_delay_ms": "300.0", "max_shots": "0", "shot_period_s": "3.0",
    "negative_fraction": "0.3", "record": "", "seed": "0", "randomize": "false",
    "min_confidence": "0.6", "consecutive": "2", "locomotion": "none", "move_vy": "0.3", "hold_s": "0.6",
}


def generate_launch_description():
    bridge = Node(package="reflex_ros", executable="sim_bridge", output="screen", parameters=[{
        "actuation_delay_ms": PV(LC("actuation_delay_ms"), value_type=float),
        "max_shots": PV(LC("max_shots"), value_type=int),
        "shot_period_s": PV(LC("shot_period_s"), value_type=float),
        "negative_fraction": PV(LC("negative_fraction"), value_type=float),
        "record": PV(LC("record"), value_type=str), "seed": PV(LC("seed"), value_type=int),
        "randomize": PV(LC("randomize"), value_type=bool), "mode": LC("mode"), "locomotion": LC("locomotion"),
    }])
    perception = Node(package="reflex_ros", executable="perception", output="screen", parameters=[{
        "use_sim_time": True, "detector": LC("detector"), "model": LC("model"), "device": LC("device"),
    }])
    behavior = Node(package="reflex_ros", executable="behavior", output="screen", parameters=[{
        "use_sim_time": True, "mode": LC("mode"), "reaction": LC("reaction"),
        "min_confidence": PV(LC("min_confidence"), value_type=float),
        "move_vy": PV(LC("move_vy"), value_type=float), "hold_s": PV(LC("hold_s"), value_type=float),
        "consecutive": PV(LC("consecutive"), value_type=int),
    }])
    return LaunchDescription(
        [DeclareLaunchArgument(k, default_value=v) for k, v in ARGS.items()]
        + [bridge, perception, behavior,
           RegisterEventHandler(OnProcessExit(target_action=bridge, on_exit=[Shutdown()]))])
