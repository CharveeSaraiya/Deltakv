#!/usr/bin/env bash
# Install Sparse-vLLM / DeltaKV (ResidualKV) env. Run on a Linux + NVIDIA GPU host.
# Deadline target: Nov 7

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/00_env.sh"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniconda/Anaconda first, then re-run."
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. ResidualKV needs an NVIDIA GPU + CUDA."
  exit 1
fi

nvidia-smi

conda create -n svllm python=3.10 -y
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate svllm

cd "$DELTAKV_REPO"

pip install torch==2.8.0 transformers[torch]==4.53.3 accelerate deepspeed==0.15.4 \
  torchvision datasets==4.1.0 bitsandbytes
pip install fire matplotlib seaborn wandb loguru ansible huggingface_hub
MAX_JOBS=8 pip install flash-attn --no-build-isolation
pip install -e .

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
import deltakv, sparsevllm
print("deltakv/sparsevllm import OK")
PY

echo "Install complete. Next: bash runs/scripts/02_download_checkpoints.sh"
