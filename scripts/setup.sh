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
