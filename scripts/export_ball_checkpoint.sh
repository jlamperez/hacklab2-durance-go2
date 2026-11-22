#!/usr/bin/env bash
# Export a ball-detector checkpoint into everything the KV260 path needs, with a PROVENANCE.md.
#
#   pixi run --manifest-path tools/river/pixi.toml export-ball CHECKPOINT.pt [OUT_DIR]
#       [--script train_ball_size.py] [--commit SHA]
#
# Runs the training code's own exporter (`<script> export`, branch feat/training) on
# CHECKPOINT.pt, so nothing is re-implemented here. Writes to OUT_DIR (default
# models/ball): the checkpoint, <prefix>_model.json, <prefix>_parameters.json/.txt,
# <prefix>_readout.json, export.log and PROVENANCE.md (commands, versions, sha256 of every
# input and output). The prefix is the one the training script exports with ("ball",
# "ball_small", ...). The bitstream is not produced: riversdk.dev makes it from
# <prefix>_model.json.
#
#   --script  training script in Training/ (default train_ball_size.py; the small network
#             is train_ball_small.py)
#   --commit  training code version (default cc540f2, the one models/ball was exported with)
#
# Runs inside the pixi environment of tools/river/pixi.toml (riverops, torch CPU).
set -euo pipefail

usage="usage: export-ball CHECKPOINT.pt [OUT_DIR] [--script NAME.py] [--commit SHA]"
SCRIPT_NAME=train_ball_size.py
COMMIT=cc540f226a95fda98308303570cdddbdeb9a9fce
POS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --script) SCRIPT_NAME=${2:?$usage}; shift 2 ;;
        --commit) COMMIT=${2:?$usage}; shift 2 ;;
        -*) echo "$usage" >&2; exit 2 ;;
        *) POS+=("$1"); shift ;;
    esac
done
CKPT=${POS[0]:?$usage}
OUT=${POS[1]:-models/ball}
WHEEL=third_party/riverops-0.2.0-py3-none-any.whl   # what tools/river/pixi.toml installs

python -c "import riverops" 2>/dev/null || {
    echo "riverops not importable: run through 'pixi run --manifest-path tools/river/pixi.toml export-ball'" >&2
    exit 1
}
REPO=$(git rev-parse --show-toplevel)
cd "$REPO"
COMMIT=$(git rev-parse --verify "$COMMIT^{commit}")
CKPT=$(realpath "$CKPT")
CKPT_NAME=$(basename "$CKPT")
mkdir -p "$OUT"
OUT=$(realpath "$OUT")
WORK=$(mktemp -d)
cleanup() { git -C "$REPO" worktree remove --force "$WORK/src" 2>/dev/null || true; rm -rf "$WORK"; }
trap cleanup EXIT

echo "== training code: Training/$SCRIPT_NAME at $COMMIT"
git worktree add --detach "$WORK/src" "$COMMIT" >/dev/null
SCRIPT="$WORK/src/Training/$SCRIPT_NAME"
test -f "$SCRIPT" || { echo "Training/$SCRIPT_NAME does not exist at $COMMIT" >&2; exit 1; }

echo "== export"
mkdir -p "$WORK/run"
cp "$CKPT" "$WORK/run/$CKPT_NAME"
(cd "$WORK/run" && python "$SCRIPT" export --checkpoint "$CKPT_NAME") 2>&1 | tee "$WORK/run/export.log"

MODELS=("$WORK"/run/*_model.json)
[ ${#MODELS[@]} -eq 1 ] && [ -f "${MODELS[0]}" ] || { echo "expected one *_model.json, got: ${MODELS[*]}" >&2; exit 1; }
PREFIX=$(basename "${MODELS[0]}" _model.json)
FILES=("$CKPT_NAME" "${PREFIX}_model.json" "${PREFIX}_parameters.json" "${PREFIX}_parameters.txt"
       "${PREFIX}_readout.json" export.log)
for f in "${FILES[@]}"; do test -f "$WORK/run/$f" || { echo "export did not write $f" >&2; exit 1; }; done

# What happens to files already in OUT_DIR, recorded before anything is overwritten.
CHANGES=""
for f in "${FILES[@]}"; do
    if [ ! -e "$OUT/$f" ]; then state="new"
    elif cmp -s "$OUT/$f" "$WORK/run/$f"; then state="unchanged (identical file was already there)"
    else state="replaced (a different file was there)"; fi
    CHANGES+="| \`$f\` | $state |"$'\n'
done
for f in "${FILES[@]}"; do cp "$WORK/run/$f" "$OUT/$f"; done

rel() { realpath --relative-to="$REPO" "$1"; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
SUMS=""
for f in "${FILES[@]}"; do SUMS+="| \`$f\` | \`$(sha "$OUT/$f")\` |"$'\n'; done
CKPT_INFO=$(python - "$OUT/$CKPT_NAME" <<'EOF'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
m = ck.get("validation_metrics", {})
window = ck.get("window_ms")
print(f"- architecture: `{ck.get('architecture')}`, epoch {ck.get('epoch')}, time steps {ck.get('time_steps')}")
print(f"- event window per frame: {window:g} ms" if window is not None else
      "- event window per frame: not stored in the checkpoint; the training code then assumes 1 ms")
if "classes" in ck:      # zone classifier (train_ball_zones.py)
    recall = ", ".join(f"{k} {v:.3f}" for k, v in m.get("recall", {}).items())
    frames = ", ".join(f"{k} {v}" for k, v in m.get("frames", {}).items())
    print(f"- classes: {', '.join(ck['classes'])}")
    print(f"- validation: balanced accuracy {m.get('balanced_accuracy', float('nan')):.3f}, "
          f"accuracy {m.get('accuracy', float('nan')):.3f}, loss {m.get('loss', float('nan')):.3f}")
    print(f"- recall per class: {recall}")
    print(f"- validation frames per class: {frames}")
else:                   # position heatmap (train_ball_size.py, train_ball_small.py)
    print(f"- validation: median error {ck.get('validation_error', float('nan')):.2f} px, "
          f"score {ck.get('validation_score', float('nan')):.3f}, found {m.get('found', float('nan')):.3f}, "
          f"false alarms {m.get('false_alarms', float('nan')):.3f} "
          f"({m.get('ball_frames')} ball / {m.get('no_ball_frames')} no-ball frames)")
if "train_int8" in ck:
    print(f"- int8 fake-quantized training (QAT): {ck['train_int8']}")
EOF
)
VERSIONS=$(python -c "
import platform
from importlib.metadata import version
print(f'   - python {platform.python_version()}')
for p in ('riverops', 'torch', 'numpy', 'pydantic', 'h5py'):
    print(f'   - {p} {version(p)}')
")
REPO_COMMIT=$(git rev-parse --short HEAD)

cat > "$OUT/PROVENANCE.md" <<EOF
# Provenance of the ball detector files

Generated by \`scripts/export_ball_checkpoint.sh\` (repo commit $REPO_COMMIT) on
$(date -u +%Y-%m-%dT%H:%M:%SZ). Every file in the table below comes from one checkpoint,
\`$CKPT_NAME\`, through the training code's own exporter. Nothing is edited by hand.

## How

\`\`\`bash
pixi run --manifest-path tools/river/pixi.toml export-ball $(rel "$CKPT") $(rel "$OUT") --script $SCRIPT_NAME --commit ${COMMIT:0:12}
\`\`\`

1. Training code: \`Training/$SCRIPT_NAME\` at commit \`$COMMIT\`
   (branch \`feat/training\`), checked out in a temporary worktree.
2. Environment: pixi, \`tools/river/pixi.toml\` (locked in \`tools/river/pixi.lock\`), with
   \`$WHEEL\` (sha256 \`$(sha "$REPO/$WHEEL")\`, Durance proprietary, not in git).
   Versions:
$VERSIONS
3. Command, run next to the checkpoint:
   \`python $SCRIPT_NAME export --checkpoint $CKPT_NAME\`.
   Its full output is \`export.log\`.

## The checkpoint

$CKPT_INFO

## Files

| File | What it is | Where it goes |
|---|---|---|
| \`$CKPT_NAME\` | the trained model (float weights, BatchNorm, readout parameters, metrics, event window) | stays here; source of everything else |
| \`${PREFIX}_model.json\` | River Model Graph (\`export_for_river\`) | riversdk.dev firmware job -> bitstream; input of \`spleat-bundle\` |
| \`${PREFIX}_parameters.json\` | BatchNorm folded, fixed point (\`export_params\`) | input of \`spleat-bundle\` |
| \`${PREFIX}_parameters.txt\` | the same values, one per line, signed as riverops writes them | not for the board: \`spleat-bundle\` writes \`weights.txt\` in two's complement (\`docs/fpga-zones3-2026-11-22.md\`) |
| \`${PREFIX}_readout.json\` | readout constants (scales, classes or channel layout) | off-chip decoding of the network output |
| \`export.log\` | output of the export command | stays here |

Board files (\`weights.txt\`, \`config.yaml\`, \`metavision_camera.json\`):
\`pixi run spleat-bundle $(rel "$OUT")/${PREFIX}_model.json $(rel "$OUT")/${PREFIX}_parameters.json\`.

## What this export did to the folder

| File | |
|---|---|
$CHANGES
## sha256

| File | sha256 |
|---|---|
$SUMS
EOF

echo "== wrote $OUT: ${FILES[*]} PROVENANCE.md"
