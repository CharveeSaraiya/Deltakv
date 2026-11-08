#!/usr/bin/env bash
# Shared paths for ResidualKV / DeltaKV course runs.
# Source this on the GPU machine after adjusting ROOT if needed.

set -euo pipefail

# Project root on the GPU machine (edit if you sync elsewhere)
export RESIDUALKV_ROOT="${RESIDUALKV_ROOT:-$HOME/kv}"
export DELTAKV_REPO="${DELTAKV_REPO:-$RESIDUALKV_ROOT/DeltaKV}"

export MODEL_ROOT="${MODEL_ROOT:-$RESIDUALKV_ROOT/models}"
export CKPT_ROOT="${CKPT_ROOT:-$RESIDUALKV_ROOT/checkpoints/compressor}"
export DATA_ROOT="${DATA_ROOT:-$RESIDUALKV_ROOT/data}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$RESIDUALKV_ROOT/outputs}"

export BASE_MODEL="${BASE_MODEL:-$MODEL_ROOT/Llama-3.1-8B-Instruct}"
export DELTAKV_CKPT="${DELTAKV_CKPT:-$CKPT_ROOT/Llama-3.1-8B-Instruct-Compressor}"

export DELTAKV_OUTPUT_DIR="$OUTPUT_ROOT"
export DELTAKV_LONGBENCH_DATA_DIR="${DELTAKV_LONGBENCH_DATA_DIR:-$DATA_ROOT/LongBench}"
export DELTAKV_DATA_DIR="${DELTAKV_DATA_DIR:-$DATA_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${DELTAKV_REPO}/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$MODEL_ROOT" "$CKPT_ROOT" "$DATA_ROOT" "$OUTPUT_ROOT"

echo "RESIDUALKV_ROOT=$RESIDUALKV_ROOT"
echo "BASE_MODEL=$BASE_MODEL"
echo "DELTAKV_CKPT=$DELTAKV_CKPT"
echo "DELTAKV_LONGBENCH_DATA_DIR=$DELTAKV_LONGBENCH_DATA_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
