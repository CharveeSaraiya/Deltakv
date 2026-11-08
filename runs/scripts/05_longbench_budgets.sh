#!/usr/bin/env bash
# Full ResidualKV LongBench runs at 10% and 25% attention budgets.
# Deadline target: Nov 13
#
# Usage:
#   bash runs/scripts/05_longbench_budgets.sh            # both 10 and 25
#   BUDGETS="10" bash runs/scripts/05_longbench_budgets.sh
#   FULL=1 bash runs/scripts/05_longbench_budgets.sh     # all EN tasks, no sample cap

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/00_env.sh"

BUDGETS="${BUDGETS:-10 25}"
FULL="${FULL:-0}"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate svllm
cd "$DELTAKV_REPO"

for BUDGET in $BUDGETS; do
  CONFIG="$RESIDUALKV_ROOT/runs/configs/llama31_8b_residualkv_budget${BUDGET}.json"
  RUN_TAG="final_b${BUDGET}_$(date +%m%d_%H%M)"
  OUT="$OUTPUT_ROOT/longbench/$RUN_TAG"
  mkdir -p "$OUT"

  echo "===== ResidualKV LongBench budget=${BUDGET}% -> $OUT ====="
  CMD=(
    python benchmark/long_bench/pred.py
    --model "llama31-8b-residualkv-b${BUDGET}"
    --model_path "$BASE_MODEL"
    --tokenizer_path "$BASE_MODEL"
    --ws 1
    --batch_size 1
    --backend hf
    --sparse_method deltakv
    --deltakv_checkpoint_path "$DELTAKV_CKPT"
    --temperature 0
    --hyper_param "$CONFIG"
    --output_root "$OUT"
  )

  if [[ "$FULL" != "1" ]]; then
    # Practical mid-size run for course deadline; set FULL=1 for paper-scale.
    CMD+=(--task qasper,hotpotqa,2wikimqa,multifieldqa_en,gov_report,multi_news,triviaqa,samsum,trec,lcc)
    CMD+=(--num_samples 20)
  fi

  "${CMD[@]}" 2>&1 | tee "$OUT/pred.log"
  python benchmark/long_bench/eval.py --path "$OUT" 2>&1 | tee "$OUT/eval.log" || true

  # Persist a one-line run record for the report
  {
    echo "budget_pct=$BUDGET"
    echo "config=$CONFIG"
    echo "base_model=$BASE_MODEL"
    echo "compressor=$DELTAKV_CKPT"
    echo "output=$OUT"
    echo "full=$FULL"
    echo "finished_at=$(date -Iseconds)"
  } > "$OUT/run_info.txt"
done

echo "All budget runs finished under $OUTPUT_ROOT/longbench/"
