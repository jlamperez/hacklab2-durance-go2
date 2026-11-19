# Camera side for the Kria KV260 (arm64): event source + perception, ROS 2 Humble.
# No PyTorch and no MuJoCo: enough for the no-learning detector and for the plumbing test.
#
#   docker build -f docker/kria/Dockerfile -t reflex-kria .          # from the repository root
#   docker run --rm -it --net=host --device /dev/video0 reflex-kria  # webcam as stand-in camera
#
# Derive the real image from this once Durance's camera SDK and RIVERLib are known.
FROM ros:humble-ros-base

RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-humble-rmw-cyclonedds-cpp python3-opencv python3-numpy python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ws
COPY reflex reflex
COPY ros2_ws/src/reflex_msgs ros2_ws/src/reflex_msgs
COPY ros2_ws/src/reflex_ros ros2_ws/src/reflex_ros
RUN . /opt/ros/humble/setup.sh && cd ros2_ws && \
    colcon build --packages-select reflex_msgs reflex_ros --cmake-args -DCMAKE_BUILD_TYPE=Release

# ROS_DOMAIN_ID is left at the default (0). To isolate from other ROS 2 machines on the
# network, pass the same value on both sides: docker run -e ROS_DOMAIN_ID=42 ...
ENV PYTHONPATH=/ws RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
COPY docker/kria/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "reflex_ros", "camera.launch.py", "detector:=simple"]
