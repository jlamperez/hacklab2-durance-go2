#!/usr/bin/env bash
# From the laptop: cross-compile our ROS 2 packages for the Kria and install the pipeline there.
#
#   scripts/kria/deploy.sh [root@192.168.77.1]
#
# 1. Builds reflex_msgs and reflex_ros for aarch64 with Durance's Docker image
#    (durance/go2-yocto:humble, made by `./manage.sh image` in their go2-kria folder) into
#    ros2_ws/install-yocto. unitree_api is not built: the board has it in /opt/go2-*.
# 2. Packs that, the pure-Python `reflex` package, the readout files of models/zones3 and
#    models/ball_v3 (run.sh model:=zones3|ball_v3) and
#    scripts/kria/run.sh, copies it to the board and unpacks it into /opt/reflex (replaced).
# Then, on the board: /opt/reflex/run.sh robot:=sim
set -euo pipefail
host=${1:-root@192.168.77.1}
repo=$(cd "$(dirname "$0")/../.." && pwd)
image=durance/go2-yocto:humble
cd "$repo"

docker image inspect "$image" >/dev/null 2>&1 || {
    echo "Docker image $image missing: run ./manage.sh image in Durance's go2-kria folder" >&2; exit 1; }

echo "== cross-compiling reflex_msgs, reflex_ros"
docker run --rm --platform linux/amd64 --user "$(id -u):$(id -g)" -e HOME=/tmp -e JOBS="${JOBS:-8}" \
    -v "$repo/ros2_ws:/ws" "$image" build --packages-select reflex_msgs reflex_ros

stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/reflex-kria/models/zones3" "$stage/reflex-kria/models/ball_v3"
cp -r ros2_ws/install-yocto "$stage/reflex-kria/install"
rsync -a --include='*/' --include='*.py' --exclude='*' --prune-empty-dirs reflex "$stage/reflex-kria/"
cp models/zones3/meta.json models/zones3/config.yaml "$stage/reflex-kria/models/zones3/"
cp models/ball_v3/ball_zones_readout.json models/ball_v3/config.yaml "$stage/reflex-kria/models/ball_v3/"
cp scripts/kria/run.sh "$stage/reflex-kria/run.sh"
git rev-parse HEAD > "$stage/reflex-kria/COMMIT" 2>/dev/null || true
tar -czf "$stage/reflex-kria.tar.gz" -C "$stage" reflex-kria
echo "== package: $(du -h "$stage/reflex-kria.tar.gz" | cut -f1)"

scp "$stage/reflex-kria.tar.gz" "$host:/tmp/"
ssh "$host" 'set -e; rm -rf /opt/reflex.new; mkdir -p /opt/reflex.new
    tar -xzf /tmp/reflex-kria.tar.gz -C /opt/reflex.new --strip-components=1
    rm -rf /opt/reflex; mv /opt/reflex.new /opt/reflex; chmod +x /opt/reflex/run.sh
    echo "installed in /opt/reflex (commit $(cat /opt/reflex/COMMIT 2>/dev/null))"'
echo "== on the board: ssh $host /opt/reflex/run.sh robot:=sim"
