"""Camera side: (stand-in) event camera + perception. Runs on the Kria, or on any machine.

    ros2 launch reflex_ros camera.launch.py detector:=simple device:=0

Publishes /reflex/decision for a behavior node anywhere on the same ROS 2 network.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration as LC
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue as PV

ARGS = {"detector": "simple", "model": "out/models/snn.pt", "device": "0", "fps": "30.0",
        "contrast_threshold": "0.25", "calibrate_s": "3.0", "inference_device": "cpu"}


def generate_launch_description():
    camera = Node(package="reflex_ros", executable="camera_webcam", output="screen", parameters=[{
        "device": PV(LC("device"), value_type=int), "fps": PV(LC("fps"), value_type=float),
        "contrast_threshold": PV(LC("contrast_threshold"), value_type=float)}])
    perception = Node(package="reflex_ros", executable="perception", output="screen", parameters=[{
        "detector": LC("detector"), "model": LC("model"), "device": LC("inference_device"),
        "calibrate_s": PV(LC("calibrate_s"), value_type=float)}])
    return LaunchDescription([DeclareLaunchArgument(k, default_value=v) for k, v in ARGS.items()]
                             + [camera, perception])
