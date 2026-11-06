#!/usr/bin/env bash
# Evaluate the SvanM2 MoE policy: flat command battery, then stair traversals.
# Mirrors interiit_xterra/scripts/eval_hs_dhav1w.sh.
#
#   xterra_mjlab/svanm2_moe_stair_sim/eval_svanm2_moe.sh
#   xterra_mjlab/svanm2_moe_stair_sim/eval_svanm2_moe.sh --no-stairs --contact-impratio 10 --label impratio10
#   CKPT=path/to/model.pt OUT=path/to/folder xterra_mjlab/svanm2_moe_stair_sim/eval_svanm2_moe.sh
#
# Extra arguments go straight to eval_policy_svanm2.py (run it with --help for all options).
# View results: tools/launch_tensorboard.sh <OUT>
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK3="$(cd "$HERE/../.." && pwd)"

PY="${PY:-$TASK3/xterra_mjlab/.venv/bin/python}"
CKPT="${CKPT:-$TASK3/xterra_mjlab/checkpoints/svanm2_moe/model_90000.pt}"
OUT="${OUT:-$TASK3/tools/results/svanm2_eval/$(date +%Y%m%d_%H%M%S)}"

if [[ ! -e "$CKPT" ]]; then
  echo "no such checkpoint: $CKPT" >&2
  exit 2
fi

echo "[eval] checkpoint: $CKPT"
echo "[eval] output    : $OUT"

cd "$TASK3"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
  "$PY" -B "$HERE/eval_policy_svanm2.py" --checkpoint-file "$CKPT" --out "$OUT" "$@"

echo
echo "[eval] figures : $OUT/flat_battery.png, $OUT/stairs.png"
echo "[eval] tables  : $OUT/runs.csv, $OUT/summary.csv"
echo "[eval] TensorBoard: $TASK3/tools/launch_tensorboard.sh $OUT"
