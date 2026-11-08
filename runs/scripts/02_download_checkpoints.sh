#!/usr/bin/env bash
# Download Llama-3.1-8B-Instruct + matching ResidualKV compressor.
# Deadline target: Nov 7
#
# Llama-3.1-8B-Instruct is gated: run `huggingface-cli login` first and accept
# the model license on Hugging Face.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/00_env.sh"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  pip install -U "huggingface_hub[cli]"
fi

echo "Downloading base model -> $BASE_MODEL"
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
  --local-dir "$BASE_MODEL"

echo "Downloading compressor -> $DELTAKV_CKPT"
huggingface-cli download JitaiHao/Llama-3.1-8B-Instruct-Compressor \
  --local-dir "$DELTAKV_CKPT"

echo "Downloading LongBench data -> $DELTAKV_LONGBENCH_DATA_DIR"
# Official LongBench release layout expects data/*.jsonl under the root.
if [[ ! -d "$DELTAKV_LONGBENCH_DATA_DIR/data" ]]; then
  git clone --depth 1 https://github.com/THUDM/LongBench.git "$DELTAKV_LONGBENCH_DATA_DIR.tmp"
  mkdir -p "$DELTAKV_LONGBENCH_DATA_DIR"
  # Prefer the packaged data/ folder if present; otherwise keep clone as root.
  if [[ -d "$DELTAKV_LONGBENCH_DATA_DIR.tmp/data" ]]; then
    mv "$DELTAKV_LONGBENCH_DATA_DIR.tmp/data" "$DELTAKV_LONGBENCH_DATA_DIR/data"
    # Keep config files useful for reference
    cp -R "$DELTAKV_LONGBENCH_DATA_DIR.tmp/config" "$DELTAKV_LONGBENCH_DATA_DIR/config" 2>/dev/null || true
    rm -rf "$DELTAKV_LONGBENCH_DATA_DIR.tmp"
  else
    rm -rf "$DELTAKV_LONGBENCH_DATA_DIR"
    mv "$DELTAKV_LONGBENCH_DATA_DIR.tmp" "$DELTAKV_LONGBENCH_DATA_DIR"
  fi
fi

ls -lah "$BASE_MODEL" | head
ls -lah "$DELTAKV_CKPT" | head
ls -lah "$DELTAKV_LONGBENCH_DATA_DIR/data" | head

echo "Downloads complete. Next: bash runs/scripts/03_smoke_test.sh"
