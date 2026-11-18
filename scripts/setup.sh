#!/usr/bin/env bash
# Fetch third-party assets that are not committed to this repo.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party

# Unitree Go2 MuJoCo model (BSD-3-Clause), pinned to the commit this repo was tested with.
MENAGERIE_COMMIT=370c439
if [ ! -f third_party/mujoco_menagerie/unitree_go2/go2.xml ]; then
  git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git \
    third_party/mujoco_menagerie
  git -C third_party/mujoco_menagerie sparse-checkout set unitree_go2
  git -C third_party/mujoco_menagerie checkout -q "$MENAGERIE_COMMIT"
fi
echo "Go2 model ready: third_party/mujoco_menagerie/unitree_go2"

# Unitree ROS 2 message definitions and example sport client (BSD-3-Clause).
UNITREE_ROS2_COMMIT=668d1ec
if [ ! -d third_party/unitree_ros2 ]; then
  git clone https://github.com/unitreerobotics/unitree_ros2.git third_party/unitree_ros2
  git -C third_party/unitree_ros2 checkout -q "$UNITREE_ROS2_COMMIT"
fi
mkdir -p ros2_ws/src
for pkg in unitree_api unitree_go; do
  ln -sfn "../../third_party/unitree_ros2/cyclonedds_ws/src/unitree/$pkg" "ros2_ws/src/$pkg"
done
echo "Unitree messages linked into ros2_ws/src (build with: pixi run -e ros ros-build)"
