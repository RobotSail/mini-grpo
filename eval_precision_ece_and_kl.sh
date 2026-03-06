#!/bin/bash
# Run ECE calibration evaluation on best-validated checkpoints (GPUs 0-3)
# Run forward KL evaluation on best-validated checkpoints (GPUs 4-7)
# Both run in parallel

set -e

RESULTS_DIR="/mnt/nvme3n1/workspace/osilkin/mini-grpo/precision_ece_results"
mkdir -p "$RESULTS_DIR"

# Read best checkpoint paths from JSON
BEST_JSON="precision_validation_results/best_checkpoints.json"

echo "============================================================"
echo "Running ECE calibration (GPUs 0-3) + Forward KL (GPUs 4-7)"
echo "============================================================"

# --- ECE CALIBRATION (GPUs 0-3) ---
echo ""
echo ">>> Starting ECE calibration evaluations..."

# Build checkpoint list from best_checkpoints.json
CKPTS=$(python3 -c "
import json
best = json.load(open('$BEST_JSON'))
print(','.join(v['full_path'] for k, v in sorted(best.items())))
")

python eval_gsm8k_parallel.py \
    --checkpoints "$CKPTS" \
    --gpus 0,1,2,3 \
    --calibration \
    --output "$RESULTS_DIR/ece_best_checkpoints.json" &
ECE_PID=$!

# --- FORWARD KL (GPUs 4-7) ---
echo ""
echo ">>> Starting forward KL evaluations..."

# Build checkpoint-dirs for forward KL - we need parent experiment dirs
# For GRPO checkpoints, the parent is the experiment dir
# For SFT, it's already the hf_format parent
# Since eval_forward_kl_parallel uses --checkpoints in single mode, just run each one

python3 -c "
import json, subprocess, sys, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile

best = json.load(open('$BEST_JSON'))
gpu_ids = [4, 5, 6, 7]
results_dir = '$RESULTS_DIR'

def run_one(exp_key, ckpt_path, gpu_id):
    out_file = os.path.join(results_dir, f'forward_kl_{exp_key}.json')
    cmd = [
        sys.executable, 'eval_forward_kl_parallel.py',
        '--mode', 'single',
        '--checkpoint', ckpt_path,
        '--gpu', str(gpu_id),
        '--output-file', out_file,
    ]
    print(f'  [GPU {gpu_id}] {exp_key}...')
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'  [GPU {gpu_id}] FAILED {exp_key}: {result.stderr[-300:]}')
        return exp_key, None
    with open(out_file) as f:
        data = json.load(f)
    fwd_kl = list(data.values())[0]['forward_kl']
    print(f'  [GPU {gpu_id}] {exp_key}: fwd_KL={fwd_kl:.4f}')
    return exp_key, data

sorted_exps = sorted(best.items())
with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
    futures = {}
    for i, (exp_key, info) in enumerate(sorted_exps):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        future = executor.submit(run_one, exp_key, info['full_path'], gpu_id)
        futures[future] = exp_key

    all_kl = {}
    for future in as_completed(futures):
        exp_key, data = future.result()
        if data:
            all_kl[exp_key] = data

# Save combined forward KL results
combined_path = os.path.join(results_dir, 'forward_kl_all.json')
with open(combined_path, 'w') as f:
    json.dump(all_kl, f, indent=2)
print(f'Saved combined forward KL to: {combined_path}')
" &
KL_PID=$!

# Wait for both
echo ""
echo "Waiting for ECE and Forward KL to complete..."
wait $ECE_PID
echo "ECE calibration complete!"
wait $KL_PID
echo "Forward KL complete!"

echo ""
echo "============================================================"
echo "ALL EVALUATIONS COMPLETE"
echo "Results in: $RESULTS_DIR"
echo "============================================================"
