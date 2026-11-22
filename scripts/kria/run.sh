#!/usr/bin/env bash
# On the Kria: the whole pipeline (camera -> FPGA SNN -> decision -> sport-mode request).
#
#   /opt/reflex/run.sh robot:=sim                   # the MuJoCo dog on the laptop is the robot
#   /opt/reflex/run.sh robot:=real                  # the real Go2, behavior DISABLED (logs only)
#   /opt/reflex/run.sh robot:=real enabled:=true    # the real Go2, moving
#   /opt/reflex/run.sh robot:=sim model:=ball_v3    # network: zones3 (default) or ball_v3; it must
#                                                   # be the one deployed on the FPGA
#
# Arguments go to `ros2 launch reflex_ros kria.launch.py` (see that file).
# Installed by scripts/kria/deploy.sh. Needs Durance's runtime in /opt/go2-* for unitree_api.
set -eo pipefail
root=$(cd "$(dirname "$0")" && pwd)

# 1. Camera check first, after every boot (docs/deploy-fpga.md). It cannot run while despot
#    holds the camera; then the camera is already streaming.
if pgrep -f despot_send_and_forget >/dev/null; then
    echo "despot_send_and_forget is already running: camera in use, skipping despot_test_camera."
    echo "Pass despot:=false, or stop it first (killall despot_send_and_forget)."
else
    echo "Camera check (despot_test_camera)..."
    if ! despot_test_camera 2>&1 | tee /tmp/despot_test_camera.log | grep -q "successfully passed"; then
        echo "despot_test_camera did not pass; see /tmp/despot_test_camera.log" >&2
        exit 1
    fi
    echo "Camera test successfully passed."
fi

# 2. FPGA server. With camera_tap (default; tap:=false to skip) the server listens on 3334 and
#    camera_tap takes 3333, where despot sends; without it the server is on 3333.
cfg=/root/.config/spleat-server/config.yaml
want=3334; case " $* " in *" tap:=false "*) want=3333 ;; esac
if ! grep -qE "port: ${want}\s*$" "$cfg"; then
    echo "spleat-server: setting port $want in $cfg and restarting"
    sed -i -E "s/port: [0-9]+/port: $want/" "$cfg"
    systemctl restart spleat-server; sleep 3
fi
systemctl is-active --quiet spleat-server || { echo "spleat-server is not running" >&2; exit 1; }

# 3. ROS: board's Humble, Durance's overlay (unitree_api messages), ours, and the reflex package.
export COLCON_PYTHON_EXECUTABLE=/usr/bin/python3 ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}
source /opt/ros/humble/setup.bash
go2=$(ls -dt /opt/go2-*/ 2>/dev/null | while read -r d; do
          [ -d "$d/share/unitree_api" ] && echo "$d" && break; done)
[ -n "$go2" ] || { echo "no /opt/go2-* with unitree_api (Durance's runtime)" >&2; exit 1; }
source "$go2/local_setup.bash"
source "$root/install/local_setup.bash"
export PYTHONPATH="$root:$PYTHONPATH"
echo "unitree_api from $go2, reflex from $root"

# 4. Which network's readout: model:=NAME -> $root/models/NAME (zones3 by default).
model=zones3 args=()
for a in "$@"; do
    case "$a" in model:=*) model=${a#model:=} ;; *) args+=("$a") ;; esac
done
[ -d "$root/models/$model" ] || { echo "no $root/models/$model" >&2; exit 1; }
want_out=$(sed -n '/^outputs:/,/^[a-z]/s/^ *channels: *//p' "$root/models/$model/config.yaml")
have_out=$(sed -n '/^outputs:/,/^[a-z]/s/^ *channels: *//p' "$cfg")
if [ "$want_out" != "$have_out" ]; then
    echo "model:=$model has $want_out outputs, the FPGA's config.yaml has $have_out: deploy that model first" >&2
    exit 1
fi
echo "network: $model ($have_out outputs)"

# 5. The model's own defaults (models/NAME/run_args, one name:=value per line); an argument
#    given on the command line wins.
if [ -f "$root/models/$model/run_args" ]; then
    while read -r line; do
        case "$line" in ""|\#*) continue ;; esac
        key=${line%%:=*}
        case " ${args[*]} " in *" $key:="*) ;; *) args+=("$line") ;; esac
    done < "$root/models/$model/run_args"
fi
echo "launch arguments: ${args[*]}"

exec ros2 launch reflex_ros kria.launch.py model_dir:="$root/models/$model" "${args[@]}"
