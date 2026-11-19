#!/usr/bin/env bash
# Recorded sessions -> training set -> fine-tune the pretrained SNN -> report.
#
#   pixi run finetune-real                          # all sessions in data/real
#   pixi run finetune-real --val-session sat_pm     # hold a whole session out (the honest number)
#   EPOCHS=30 LR=1e-3 INIT=out/models/snn.pt pixi run finetune-real
#
# Result: out/models/snn_real.pt (+ .json with the metrics). The pretrained model is not touched.
set -euo pipefail
cd "$(dirname "$0")/.."
init=${INIT:-out/models/snn.pt}
start=$(date +%s)
python -m reflex.data.recording data/real/*/ --out data/real_ds
python -m reflex.train --data data/real_ds --init "$init" --out out/models/snn_real.pt \
  --epochs "${EPOCHS:-15}" --lr "${LR:-5e-4}" "$@"
echo "total: $(( $(date +%s) - start )) s. Use it with: detector:=snn model:=out/models/snn_real.pt"
