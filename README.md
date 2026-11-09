# ResidualKV lab package (Charvee / Team 6)

Layout matches Anees’s `charvee_residualkv.zip` for the SJSU RTX 5090 lab.

```
298a-kv-cache-compression/   # team repo: harness, 800-record data, members/charvee runner
DeltaKV/                     # authors’ code (use commit 11a837b)
charvee_residualkv.zip       # original handoff archive
```

## Lab (Docker container)

```bash
cd /app/298a-kv-cache-compression
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
pip install transformers==5.2.0 rouge rapidfuzz loguru
hf auth login

# smoke: 2 records/task
python members/charvee/run_residualkv.py --compression-ratio 0.9 --per-task 2

# full budgets (priority order)
nohup bash -c 'for r in 0.9 0.75 0.8 0.95; do python members/charvee/run_residualkv.py --compression-ratio $r; done' > residualkv.log 2>&1 &
tail -f residualkv.log
```

Results: `298a-kv-cache-compression/results/residualkv/longbench/`  
Data: `298a-kv-cache-compression/data/processed_longbench/processed_longbench_dev.jsonl` (800 records)

Do not change data, prompts, scoring, configs, or ratios.
