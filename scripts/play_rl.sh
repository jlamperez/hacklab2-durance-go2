#!/usr/bin/env bash
# Watch the Go2 walking policy trained with unitree_rl_mjlab (optional extra; it does
# not touch this project's simulator). Uses the newest checkpoint unless one is given.
#
#   pixi run rl-play                       # 3D window, one robot
#   pixi run rl-play --num_envs 9          # nine robots, each with its own velocity command
#   pixi run rl-play --viewer viser        # in the browser at http://localhost:8080
set -euo pipefail
cd "$(dirname "$0")/../third_party/unitree_rl_mjlab"
ckpt=$(ls -v logs/rsl_rl/go2_velocity/*/model_*.pt 2>/dev/null | tail -1 || true)
[ -n "$ckpt" ] || { echo "no checkpoint; train first (see docs/go2-control.md)"; exit 1; }
echo "checkpoint: $ckpt"
WANDB_MODE=disabled MUJOCO_GL=glfw .venv/bin/python scripts/play.py Unitree-Go2-Flat \
  --checkpoint_file="$ckpt" --num_envs 1 "$@"
