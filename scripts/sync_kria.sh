#!/usr/bin/env bash
# Copy what the Kria needs (reflex package, ROS packages, Docker recipe) and rebuild its image.
#
#   scripts/sync_kria.sh [user@host]      # default: ubuntu@kria.local
#
# Patterns start with / so they only match at the repository root: a bare "data" would
# also drop reflex/data.
set -euo pipefail
target=${1:-ubuntu@kria.local}
cd "$(dirname "$0")/.."
rsync -az --delete \
  --exclude=/.git --exclude=/.pixi --exclude=/out --exclude=/data --exclude=/third_party --exclude=/models \
  --exclude=/tests --exclude=/docs --exclude=/ros2_ws/build --exclude=/ros2_ws/install --exclude=/ros2_ws/log \
  --exclude=/ros2_ws/src/unitree_api --exclude=/ros2_ws/src/unitree_go --exclude='__pycache__' --exclude=/build.log \
  ./ "$target":hacklab2-durance-go2/
# --network=host: on some boards the default Docker bridge has no working DNS during builds.
ssh "$target" 'cd hacklab2-durance-go2 && docker build --network=host --pull=false -q -f docker/kria/Dockerfile -t reflex-kria .'
echo "run it with:  ssh -t $target docker run --rm -it --net=host --device /dev/video0 reflex-kria"
