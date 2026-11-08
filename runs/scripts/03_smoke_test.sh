#!/usr/bin/env bash
# ResidualKV smoke test on Llama-3.1-8B-Instruct.
# Deadline target: Nov 7

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/00_env.sh"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate svllm

cd "$DELTAKV_REPO"

if [[ ! -d "$BASE_MODEL" ]]; then
  echo "Missing base model: $BASE_MODEL"
  exit 1
fi
if [[ ! -d "$DELTAKV_CKPT" ]]; then
  echo "Missing compressor checkpoint: $DELTAKV_CKPT"
  exit 1
fi

SMOKE_OUT="$OUTPUT_ROOT/smoke/$(date +%m%d_%H%M)"
mkdir -p "$SMOKE_OUT"

# Prefer Sparse-vLLM throughput smoke when CUDA kernels are available.
# Falls back to a tiny HF LongBench sample if sparsevllm microbench fails.
set +e
PYTHONPATH="$DELTAKV_REPO/src" python scripts/benchmarks/bench_sparse_vllm.py \
  --model_path "$BASE_MODEL" \
  --lengths 1024 \
  --batch_sizes 1 \
  --methods deltakv \
  --output_len 4 \
  --temperature 0 \
  --hyper_params "{\"gpu_memory_utilization\":0.85,\"engine_prefill_chunk_size\":512,\"max_num_seqs_in_batch\":1,\"max_decoding_seqs\":1,\"max_num_batched_tokens\":2048,\"full_attention_layers\":\"0,1\",\"sink_keep_tokens\":4,\"recent_keep_tokens\":32,\"decode_keep_tokens\":64,\"prefill_keep_tokens\":64,\"deltakv_checkpoint_path\":\"$DELTAKV_CKPT\",\"deltakv_center_ratio\":0.1,\"deltakv_neighbor_count\":1,\"deltakv_latent_dim\":512,\"deltakv_latent_quant_bits\":4,\"full_layer_kv_quant_bits\":4,\"enable_full_layer_kivi_quant\":true,\"deltakv_full_pool_reserve_ratio\":0.2}" \
  2>&1 | tee "$SMOKE_OUT/sparsevllm_smoke.log"
SVLLM_RC=${PIPESTATUS[0]}
set -e

if [[ $SVLLM_RC -ne 0 ]]; then
  echo "Sparse-vLLM smoke failed (exit $SVLLM_RC). Trying HF LongBench 1-sample smoke..."
  python benchmark/long_bench/pred.py \
    --model llama31-8b-residualkv-smoke \
    --model_path "$BASE_MODEL" \
    --tokenizer_path "$BASE_MODEL" \
    --ws 1 \
    --batch_size 1 \
    --backend hf \
    --sparse_method deltakv \
    --deltakv_checkpoint_path "$DELTAKV_CKPT" \
    --task qasper \
    --num_samples 1 \
    --temperature 0 \
    --hyper_param "$RESIDUALKV_ROOT/runs/configs/llama31_8b_residualkv_budget25.json" \
    2>&1 | tee "$SMOKE_OUT/hf_smoke.log"
fi

echo "Smoke artifacts in: $SMOKE_OUT"
echo "Look for compressor load lines in the log (checkpoint path must appear)."
