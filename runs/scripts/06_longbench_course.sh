#!/usr/bin/env bash
# Course LongBench (required):
#   - all records (no --num_samples / no --max-prompt-tokens)
#   - compression ratios 0.75, 0.8, 0.9, 0.95  => keep 25%, 20%, 10%, 5%
#   - if GPU time is short: run 0.75 and 0.9 first
#   - report kv_cache_gb for each run
#   - compare against shared baseline (do NOT rerun baseline):
#       results/longbench/Llama-3.1-8B-Instruct_none_0.0.jsonl
#
# Usage:
#   # Priority pair first
#   RATIOS="0.75 0.90" bash runs/scripts/06_longbench_course.sh
#   # Then remaining
#   RATIOS="0.80 0.95" bash runs/scripts/06_longbench_course.sh
#   # Or all four
#   bash runs/scripts/06_longbench_course.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/00_env.sh"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate svllm
cd "$DELTAKV_REPO"

METHOD_NAME="${METHOD_NAME:-residualkv}"
MODEL_TAG="${MODEL_TAG:-Llama-3.1-8B-Instruct}"
# Priority default: 0.75 then 0.90; override with RATIOS="0.75 0.80 0.90 0.95"
RATIOS="${RATIOS:-0.75 0.90}"
PROBE_SEQ_LEN="${PROBE_SEQ_LEN:-8192}"

RESULTS_DIR="${RESULTS_DIR:-$RESIDUALKV_ROOT/results/longbench}"
mkdir -p "$RESULTS_DIR" "$OUTPUT_ROOT/longbench_course"

BASELINE="${BASELINE:-$RESULTS_DIR/${MODEL_TAG}_none_0.0.jsonl}"
if [[ ! -f "$BASELINE" ]]; then
  echo "WARNING: shared baseline not found at: $BASELINE"
  echo "Place the provided file there (or set BASELINE=...), then re-run compare."
fi

ratio_to_config() {
  case "$1" in
    0.75|0.750) echo "$RESIDUALKV_ROOT/runs/configs/llama31_8b_residualkv_cr0.75.json" ;;
    0.8|0.80|0.800) echo "$RESIDUALKV_ROOT/runs/configs/llama31_8b_residualkv_cr0.80.json" ;;
    0.9|0.90|0.900) echo "$RESIDUALKV_ROOT/runs/configs/llama31_8b_residualkv_cr0.90.json" ;;
    0.95|0.950) echo "$RESIDUALKV_ROOT/runs/configs/llama31_8b_residualkv_cr0.95.json" ;;
    *) echo ""; return 1 ;;
  esac
}

fmt_ratio() {
  python3 - <<PY
r=float("$1")
print(f"{r:.2f}")
PY
}

for RAW in $RATIOS; do
  CR="$(fmt_ratio "$RAW")"
  CONFIG="$(ratio_to_config "$CR" || true)"
  if [[ -z "$CONFIG" || ! -f "$CONFIG" ]]; then
    echo "Unknown/missing config for compression_ratio=$CR"
    exit 1
  fi
  KEEP="$(python3 - <<PY
print(f"{1.0 - float('$CR'):.2f}")
PY
)"

  RUN_DIR="$OUTPUT_ROOT/longbench_course/${METHOD_NAME}_cr${CR}_$(date +%m%d_%H%M)"
  mkdir -p "$RUN_DIR"
  OUT_JSONL="$RESULTS_DIR/${MODEL_TAG}_${METHOD_NAME}_${CR}.jsonl"
  META="$RESULTS_DIR/${MODEL_TAG}_${METHOD_NAME}_${CR}.meta.json"

  echo "===== ResidualKV LongBench cr=${CR} keep=${KEEP} ====="
  echo "config=$CONFIG"
  echo "run_dir=$RUN_DIR"
  echo "out_jsonl=$OUT_JSONL"

  # 1) Measure kv_cache_gb (probe; no prompt truncation flag)
  python "$RESIDUALKV_ROOT/runs/scripts/measure_kv_cache_gb.py" \
    --model-path "$BASE_MODEL" \
    --deltakv-checkpoint-path "$DELTAKV_CKPT" \
    --hyper-param "$CONFIG" \
    --seq-len "$PROBE_SEQ_LEN" \
    --out "$RUN_DIR/kv_cache_gb.json" \
    2>&1 | tee "$RUN_DIR/kv_cache_probe.log"

  KV_CACHE_GB="$(python3 - <<PY
import json
print(json.load(open("$RUN_DIR/kv_cache_gb.json"))["kv_cache_gb"])
PY
)"

  # 2) Full LongBench English suite: NO --num_samples, NO --max-prompt-tokens
  #    (pred.py has no max-prompt-tokens flag; do not add truncation yourself)
  python benchmark/long_bench/pred.py \
    --model "${MODEL_TAG}_${METHOD_NAME}_${CR}" \
    --model_path "$BASE_MODEL" \
    --tokenizer_path "$BASE_MODEL" \
    --ws 1 \
    --batch_size 1 \
    --backend hf \
    --sparse_method deltakv \
    --deltakv_checkpoint_path "$DELTAKV_CKPT" \
    --temperature 0 \
    --hyper_param "$CONFIG" \
    --output_root "$RUN_DIR" \
    2>&1 | tee "$RUN_DIR/pred.log"

  python benchmark/long_bench/eval.py --path "$RUN_DIR" 2>&1 | tee "$RUN_DIR/eval.log"

  # 3) Merge per-task jsonl -> course-style single file + attach kv_cache_gb
  python3 - <<PY
import json, glob, os
from pathlib import Path

run_dir = Path("$RUN_DIR")
out_jsonl = Path("$OUT_JSONL")
rows = []
for path in sorted(run_dir.glob("*.jsonl")):
    if path.name in {"raw_outputs.jsonl", "parsed_outputs.jsonl", "sample_results.jsonl", "longbench_mini_selection.jsonl"}:
        continue
    task = path.stem
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row.setdefault("dataset", task)
            row.setdefault("task", task)
            row["compression_ratio"] = float("$CR")
            row["keep_ratio"] = float("$KEEP")
            row["method"] = "$METHOD_NAME"
            row["kv_cache_gb"] = float("$KV_CACHE_GB")
            rows.append(row)

out_jsonl.parent.mkdir(parents=True, exist_ok=True)
with out_jsonl.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

metrics = {}
result_path = run_dir / "result.json"
if result_path.exists():
    metrics = json.loads(result_path.read_text(encoding="utf-8"))

meta = {
    "model": "$MODEL_TAG",
    "method": "$METHOD_NAME",
    "compression_ratio": float("$CR"),
    "keep_ratio": float("$KEEP"),
    "kv_cache_gb": float("$KV_CACHE_GB"),
    "n_records": len(rows),
    "run_dir": str(run_dir),
    "predictions": str(out_jsonl),
    "metrics": metrics,
    "baseline": "$BASELINE",
    "note": "No --max-prompt-tokens / no sample cap. Compare vs shared none_0.0 baseline.",
}
Path("$META").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(json.dumps({"wrote": str(out_jsonl), "n_records": len(rows), "kv_cache_gb": float("$KV_CACHE_GB")}, indent=2))
PY

  echo "Saved $OUT_JSONL | kv_cache_gb=$KV_CACHE_GB | records merged from $RUN_DIR"
done

echo
echo "Done. Course artifacts under: $RESULTS_DIR"
ls -lah "$RESULTS_DIR"/${MODEL_TAG}_${METHOD_NAME}_*.jsonl 2>/dev/null || true
ls -lah "$RESULTS_DIR"/${MODEL_TAG}_${METHOD_NAME}_*.meta.json 2>/dev/null || true

if [[ -f "$BASELINE" ]]; then
  echo "Baseline present: $BASELINE (do not rerun). Compare scores in *.meta.json vs baseline eval."
else
  echo "Remember to copy shared baseline to: $BASELINE"
fi
