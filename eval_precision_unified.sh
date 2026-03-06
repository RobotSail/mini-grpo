#!/bin/bash
# Run unified evaluation (accuracy + forward KL + ECE) on all 12 best-validated checkpoints
# Distributes across GPUs using subprocess parallelism

set -e

BEST_JSON="precision_validation_results/best_checkpoints.json"
RESULTS_DIR="precision_unified_results"
N_RUNS="${1:-3}"
GPUS="${2:-0,1,2,3,4,5,6,7}"

mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "Unified Evaluation: Accuracy (${N_RUNS}x) + Forward KL + ECE"
echo "GPUs: $GPUS"
echo "============================================================"

python3 -c "
import json, subprocess, sys, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

best = json.load(open('$BEST_JSON'))
gpu_ids = [int(g) for g in '$GPUS'.split(',')]
n_runs = $N_RUNS
results_dir = '$RESULTS_DIR'

def run_one(exp_key, ckpt_path, gpu_id):
    out_file = os.path.join(results_dir, f'{exp_key}.json')
    cmd = [
        sys.executable, 'eval_unified.py',
        '--checkpoint', ckpt_path,
        '--gpu', str(gpu_id),
        '--n-runs', str(n_runs),
        '--output', out_file,
    ]
    print(f'[GPU {gpu_id}] Starting {exp_key}...')
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'[GPU {gpu_id}] FAILED {exp_key}')
        print(result.stderr[-500:])
        return exp_key, None
    data = json.load(open(out_file))
    print(f'[GPU {gpu_id}] {exp_key}: acc={data[\"accuracy_mean\"]:.2%} fwd_kl={data[\"forward_kl\"]:.4f} ece={data[\"ece\"]:.4f}')
    return exp_key, data

sorted_exps = sorted(best.items())
with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
    futures = {}
    for i, (exp_key, info) in enumerate(sorted_exps):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        future = executor.submit(run_one, exp_key, info['full_path'], gpu_id)
        futures[future] = exp_key

    all_results = {}
    for future in as_completed(futures):
        exp_key, data = future.result()
        if data:
            all_results[exp_key] = data

# Save combined
combined_path = os.path.join(results_dir, 'all_results.json')
with open(combined_path, 'w') as f:
    json.dump(all_results, f, indent=2)

# Print summary
print()
print('=' * 80)
print(f'{\"Experiment\":25s} {\"Acc Mean\":>9s} {\"Acc Std\":>8s} {\"Fwd KL\":>8s} {\"ECE\":>7s} {\"Conf\":>7s}')
print('-' * 80)
for k in sorted(all_results):
    r = all_results[k]
    print(f'{k:25s} {r[\"accuracy_mean\"]:8.2%} {r[\"accuracy_std\"]:7.2%} {r[\"forward_kl\"]:7.4f} {r[\"ece\"]:6.4f} {r[\"mean_confidence\"]:6.4f}')
print('=' * 80)
print(f'Saved: {combined_path}')
"

echo ""
echo "Done! Results in: $RESULTS_DIR"
