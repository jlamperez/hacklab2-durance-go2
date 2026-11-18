#!/usr/bin/env bash
# Optional: set up Unitree's RL trainer (unitree_rl_mjlab) to train a Go2 walking policy.
# It lives in its own virtualenv because it needs older MuJoCo and Warp than this repo.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party
DIR=third_party/unitree_rl_mjlab
COMMIT=1425b15
if [ ! -d "$DIR" ]; then
  git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git "$DIR"
  git -C "$DIR" checkout -q "$COMMIT"
fi
cd "$DIR"
# The repo does not pin its transitive dependencies and current releases no longer work
# together (MuJoCo 4, Warp 1.19, tyro). Freeze resolution three days after mjlab 1.2.0.
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python --exclude-newer 2026-03-09T00:00:00Z -e . scipy
cat <<'MSG'
Ready. Train with (about 2 h for the full 10001 iterations on an RTX 5090, checkpoint every 100):
  cd third_party/unitree_rl_mjlab
  WANDB_MODE=disabled MUJOCO_GL=egl .venv/bin/python scripts/train.py Unitree-Go2-Flat --env.scene.num-envs=4096
Checkpoints: logs/rsl_rl/go2_velocity/<date>/model_<iteration>.pt
MSG
