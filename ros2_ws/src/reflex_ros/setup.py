from glob import glob

from setuptools import setup

package_name = "reflex_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "perception = reflex_ros.perception_node:main",
            "behavior = reflex_ros.behavior_node:main",
            "sim_bridge = reflex_ros.sim_bridge_node:main",
            "camera_webcam = reflex_ros.camera_webcam_node:main",
            "cmd_vel_to_sport = reflex_ros.cmd_vel_to_sport_node:main",
            "recorder = reflex_ros.recorder_node:main",
            "tricks = reflex_ros.tricks_node:main",
            "web_viewer = reflex_ros.web_viewer_node:main",
            "target = reflex_ros.target_node:main",
            "ball_target = reflex_ros.ball_target_node:main",
            "ball_zone_sim = reflex_ros.ball_zone_sim_node:main",
            "follow = reflex_ros.follow_node:main",
            "event_camera = reflex_ros.event_camera_node:main",
        ],
    },
)
