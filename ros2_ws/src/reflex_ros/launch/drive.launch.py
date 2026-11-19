"""Drive the simulated dog over ROS 2 with a keyboard (or anything that publishes /cmd_vel).

    ros2 launch reflex_ros drive.launch.py
    ros2 run teleop_twist_keyboard teleop_twist_keyboard     # in a second terminal

/cmd_vel -> cmd_vel_to_sport -> Move on /api/sport/request -> walking dog in MuJoCo.
On the real robot, run cmd_vel_to_sport alone: the Go2 takes the simulator's place.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration as LC
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue as PV

ARGS = {"actuation_delay_ms": "300.0", "viewer": "true", "timeout_s": "0.0"}


def generate_launch_description():
    bridge = Node(package="reflex_ros", executable="sim_bridge", output="screen", parameters=[{
        "robot_only": True, "locomotion": "rl", "viewer": PV(LC("viewer"), value_type=bool),
        "actuation_delay_ms": PV(LC("actuation_delay_ms"), value_type=float)}],
        additional_env={"MUJOCO_GL": "glfw"})
    teleop = Node(package="reflex_ros", executable="cmd_vel_to_sport", output="screen", parameters=[{
        "timeout_s": PV(LC("timeout_s"), value_type=float)}])
    return LaunchDescription([DeclareLaunchArgument(k, default_value=v) for k, v in ARGS.items()]
                             + [bridge, teleop,
                                RegisterEventHandler(OnProcessExit(target_action=bridge, on_exit=[Shutdown()]))])
