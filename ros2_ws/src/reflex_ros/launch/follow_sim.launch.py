"""Follow a ball in simulation over ROS 2: the same nodes as on the real robot.

    ros2 launch reflex_ros follow_sim.launch.py          # then, second terminal: ros-follow, press f
    ros2 launch reflex_ros follow_sim.launch.py follow:=true    # follow node included, starts by itself

  sim_bridge (robot only, walking dog, its event camera, a ball moved by script)
    -> /reflex/events_window -> target node (CentroidTracker) -> /reflex/target
    -> follow node -> Move / StopMove on /api/sport/request -> walking dog

Real robot: the camera side publishes /reflex/events_window (or the SNN adapter publishes
/reflex/target directly), the Go2 replaces sim_bridge, and the follow node runs unchanged.
Keep the scene still for the first seconds: the target node calibrates.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, Shutdown
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration as LC
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue as PV

ARGS = {"actuation_delay_ms": "300.0", "viewer": "true", "scene": "lab", "follow": "false", "mode": "walk",
        "target_start_s": "5.0", "zone_top": "0.3"}


def generate_launch_description():
    bridge = Node(package="reflex_ros", executable="sim_bridge", output="screen", parameters=[{
        "robot_only": True, "locomotion": "rl", "camera": True, "target_path": True,
        "target_start_s": PV(LC("target_start_s"), value_type=float),
        "viewer": PV(LC("viewer"), value_type=bool), "scene": LC("scene"),
        "actuation_delay_ms": PV(LC("actuation_delay_ms"), value_type=float)}],
        additional_env={"MUJOCO_GL": "glfw"})
    target = Node(package="reflex_ros", executable="target", output="screen", parameters=[{
        "zone_top": PV(LC("zone_top"), value_type=float)}])
    follow = Node(package="reflex_ros", executable="follow", output="screen", parameters=[{
        "mode": LC("mode"), "autostart": True}], condition=IfCondition(LC("follow")))
    return LaunchDescription([DeclareLaunchArgument(k, default_value=v) for k, v in ARGS.items()]
                             + [bridge, target, follow,
                                RegisterEventHandler(OnProcessExit(target_action=bridge, on_exit=[Shutdown()]))])
