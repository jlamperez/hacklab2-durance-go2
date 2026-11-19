"""Robot side, simulated: behavior + a MuJoCo dog that only acts as the robot.

    ros2 launch reflex_ros robot_sim.launch.py viewer:=true

Listens to /reflex/decision from a camera side running here or on another machine.
For the real robot, run the behavior node alone: the Go2 takes the bridge's place.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration as LC
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue as PV

ARGS = {"mode": "keeper", "reaction": "euler", "actuation_delay_ms": "300.0", "viewer": "true",
        "min_confidence": "0.6", "consecutive": "2", "locomotion": "none", "move_vy": "0.3", "hold_s": "0.6", "enabled": "true",
        "scene": "lab"}


def generate_launch_description():
    bridge = Node(package="reflex_ros", executable="sim_bridge", output="screen", parameters=[{
        "robot_only": True, "viewer": PV(LC("viewer"), value_type=bool),
        "actuation_delay_ms": PV(LC("actuation_delay_ms"), value_type=float), "mode": LC("mode"), "locomotion": LC("locomotion"),
        "scene": LC("scene")}],
        additional_env={"MUJOCO_GL": "glfw"})
    behavior = Node(package="reflex_ros", executable="behavior", output="screen", parameters=[{
        "mode": LC("mode"), "reaction": LC("reaction"), "enabled": PV(LC("enabled"), value_type=bool),
        "min_confidence": PV(LC("min_confidence"), value_type=float),
        "move_vy": PV(LC("move_vy"), value_type=float), "hold_s": PV(LC("hold_s"), value_type=float),
        "consecutive": PV(LC("consecutive"), value_type=int)}])
    return LaunchDescription([DeclareLaunchArgument(k, default_value=v) for k, v in ARGS.items()]
                             + [bridge, behavior,
                                RegisterEventHandler(OnProcessExit(target_action=bridge, on_exit=[Shutdown()]))])
