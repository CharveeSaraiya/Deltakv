#!/usr/bin/env bash
# First LongBench "dev" result: small English subset, few samples/task.
# Deadline target: Nov 10
#
# Usage:
#   BUDGET=25 bash runs/scripts/04_longbench_dev.sh
#   BUDGET=10 bash runs/scripts/04_longbench_dev.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/00_env.sh"

BUDGET="${BUDGET:-25}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
TASKS="${TASKS:-qasper,hotpotqa,multi_news,triviaqa}"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate svllm

cd "$DELTAKV_REPO"

CONFIG="$RESIDUALKV_ROOT/runs/configs/llama31_8b_residualkv_budget${BUDGET}.json"
if [[ ! -f "$CONFIG" ]]; then
  echo "Missing config: $CONFIG (use BUDGET=10 or BUDGET=25)"
  exit 1
fi

RUN_TAG="dev_b${BUDGET}_n${NUM_SAMPLES}_$(date +%m%d_%H%M)"
OUT="$OUTPUT_ROOT/longbench/$RUN_TAG"
mkdir -p "$OUT"

echo "Running LongBench dev | budget=${BUDGET}% | tasks=$TASKS | num_samples=$NUM_SAMPLES"
echo "Config: $CONFIG"
echo "Output: $OUT"

python benchmark/long_bench/pred.py \
  --model "llama31-8b-residualkv-b${BUDGET}" \
  --model_path "$BASE_MODEL" \
  --tokenizer_path "$BASE_MODEL" \
  --ws 1 \
  --batch_size 1 \
  --backend hf \
  --sparse_method deltakv \
  --deltakv_checkpoint_path "$DELTAKV_CKPT" \
  --task "$TASKS" \
  --num_samples "$NUM_SAMPLES" \
  --temperature 0 \
  --hyper_param "$CONFIG" \
  --output_root "$OUT" \
  2>&1 | tee "$OUT/pred.log"

python benchmark/long_bench/eval.py --path "$OUT" 2>&1 | tee "$OUT/eval.log" || true

echo "Done. Predictions: $OUT"
echo "Submit/share: pred.log, eval.log, and *.jsonl under $OUT"
