#!/usr/bin/env python3
"""Probe ResidualKV / DeltaKV resident KV cache size in GB for a fixed seq length."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _tensor_nbytes(obj) -> int:
    total = 0
    if torch.is_tensor(obj):
        return int(obj.numel() * obj.element_size())
    if isinstance(obj, dict):
        for v in obj.values():
            total += _tensor_nbytes(v)
        return total
    if isinstance(obj, (list, tuple)):
        for v in obj:
            total += _tensor_nbytes(v)
        return total
    if hasattr(obj, "__dict__"):
        for v in vars(obj).values():
            total += _tensor_nbytes(v)
        return total
    return 0


def full_fp16_kv_gb(num_layers: int, num_kv_heads: int, head_dim: int, seq_len: int) -> float:
    return (2 * num_layers * num_kv_heads * head_dim * seq_len * 2) / (1024**3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--deltakv-checkpoint-path", required=True)
    parser.add_argument("--hyper-param", required=True)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    parser.add_argument("--sparse-method", default="deltakv")
    args = parser.parse_args()

    from deltakv.get_chat_api import get_generate_api
    from transformers import AutoTokenizer

    with open(args.hyper_param, encoding="utf-8") as f:
        hyper = json.load(f)
    hyper["deltakv_checkpoint_path"] = args.deltakv_checkpoint_path

    gen, model = get_generate_api(
        args.model_path,
        hyper,
        tokenizer_path=args.model_path,
        sparse_method=args.sparse_method,
        cuda_device=args.device,
        backend="hf",
        return_kv_cache=True,
        return_model=True,
    )

    cfg = model.config
    num_layers = int(cfg.num_hidden_layers)
    num_kv_heads = int(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads))
    head_dim = int(cfg.hidden_size // cfg.num_attention_heads)
    full_gb = full_fp16_kv_gb(num_layers, num_kv_heads, head_dim, args.seq_len)

    keep = float(hyper.get("decode_keep_tokens", 1.0) or 1.0)
    if keep > 1.0:
        # Absolute token budget; convert to ratio vs probe length for naive estimate only.
        keep = min(1.0, keep / float(args.seq_len))
    naive_kept_gb = full_gb * keep

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build an exact-length synthetic prompt (no max-prompt-tokens truncation).
    unit = "word "
    ids = tokenizer.encode(unit * (args.seq_len + 64), add_special_tokens=False)
    ids = ids[: args.seq_len]
    prompt = tokenizer.decode(ids)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()

    text, past = gen(prompt, max_new_tokens=1, temperature=0.0, do_sample=False)
    after = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()

    kv_bytes = _tensor_nbytes(past)
    measured_gb = kv_bytes / (1024**3) if kv_bytes > 0 else max(0.0, (after - before) / (1024**3))

    report = {
        "seq_len": args.seq_len,
        "prompt_token_count": len(ids),
        "generated_preview": text[:80] if isinstance(text, str) else "",
        "full_fp16_kv_cache_gb": round(full_gb, 6),
        "keep_ratio": keep,
        "naive_kept_fp16_kv_cache_gb": round(naive_kept_gb, 6),
        "kv_cache_gb": round(measured_gb, 6),
        "cuda_alloc_delta_gb": round((after - before) / (1024**3), 6),
        "cuda_peak_gb": round(peak / (1024**3), 6),
        "hyper_param": args.hyper_param,
        "model_path": args.model_path,
        "deltakv_checkpoint_path": args.deltakv_checkpoint_path,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
