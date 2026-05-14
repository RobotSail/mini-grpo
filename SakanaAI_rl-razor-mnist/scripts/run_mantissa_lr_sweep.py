#!/usr/bin/env python3
"""
Mantissa × LR sweep matching RL's Razor paper setup (Appendix B.3).

Sweeps 15 log-spaced LRs in [3e-6, 1e-3] × {1, 2} epochs × mantissa settings,
for each of the 5 methods. Reports KL, parity accuracy, fashion accuracy,
and weight-level metrics for each run.
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from rl_razor.model import MLP
from rl_razor.training.grpo import grpo_finetune
from rl_razor.training.sft import sft_finetune
from rl_razor.data import get_fashion_mnist, get_parity_mnist, create_dataloader
from rl_razor.utils import set_seed, snap_to_lattice

DEVICE = "cuda"
SEED = 42
PRETRAIN_PATH = "experiments/mantissa_sweep/pretrain/pretrained_model.pt"

LRS = np.logspace(np.log10(3e-6), np.log10(1e-3), 15).tolist()
EPOCHS_LIST = [1, 2]
SCHEDULERS = ["cosine_with_warmup", "constant_with_warmup"]
MANTISSA_BITS = [7, 10, 13, 16, 0]  # 0 = fp32 (no snapping)

METHODS = [
    ("grpo", {"kl_coef": 0.0}),
    ("grpo_kl", {"kl_coef": 0.1}),
    ("sft1", {}),
    ("sft2", {}),
    ("oracle", {}),
]


def compute_weight_metrics(base_model, ft_model):
    base_sd = {n: p.data for n, p in base_model.named_parameters()}
    frob_sq = 0; nz = 0; above = 0; total = 0
    for n, p in ft_model.named_parameters():
        if n in base_sd:
            delta = p.data.float() - base_sd[n].float()
            frob_sq += (delta ** 2).sum().item()
            nz += (delta != 0).sum().item()
            above += (delta.abs() >= 1e-5).sum().item()
            total += delta.numel()
    return {
        "frobenius_norm": frob_sq ** 0.5,
        "l0_exact": nz / total if total else 0,
        "l0_thresh": above / total if total else 0,
    }


def run_single(method_name, method_kwargs, lr, epochs, scheduler, mbits):
    set_seed(SEED)
    base = MLP.from_checkpoint(PRETRAIN_PATH, device=DEVICE)
    fashion_val = get_fashion_mnist(train=False)
    fashion_loader = create_dataloader(fashion_val, batch_size=64, shuffle=False)

    if method_name in ("sft1", "sft2", "oracle"):
        r = sft_finetune(
            base_model=base, label_mode=method_name,
            batch_size=64, learning_rate=lr, num_epochs=epochs,
            scheduler_type=scheduler, warmup_ratio=0.1, weight_decay=0.0,
            seed=SEED, device=DEVICE, eval_fashion=True,
            fashion_loader=fashion_loader, verbose=False,
            mantissa_bits=mbits,
        )
    else:
        kl_coef = method_kwargs.get("kl_coef", 0.0)
        r = grpo_finetune(
            base_model=base, batch_size=64, group_size=8,
            learning_rate=lr, num_epochs=epochs,
            scheduler_type=scheduler, warmup_ratio=0.1, weight_decay=0.0,
            kl_coef=kl_coef, seed=SEED, device=DEVICE,
            eval_fashion=True, fashion_loader=fashion_loader,
            verbose=False, mantissa_bits=mbits,
        )

    wm = compute_weight_metrics(base, r["model"])

    return {
        "method": method_name,
        "lr": lr,
        "epochs": epochs,
        "scheduler": scheduler,
        "mantissa_bits": mbits,
        "parity_acc": r["final_parity_acc"],
        "kl": r["final_kl_divergence"],
        "fashion_acc": r["final_fashion_acc"],
        **wm,
    }


def main():
    output_path = "experiments/mantissa_sweep/lr_sweep_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Resume from partial results if they exist
    results = []
    done = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        for r in results:
            done.add((r["method"], r["lr"], r["epochs"], r["scheduler"], r["mantissa_bits"]))
        print(f"Resuming: {len(done)} runs already complete")

    total = len(METHODS) * len(LRS) * len(EPOCHS_LIST) * len(SCHEDULERS) * len(MANTISSA_BITS)
    print(f"Total runs: {total} ({len(METHODS)} methods × {len(LRS)} LRs × "
          f"{len(EPOCHS_LIST)} epochs × {len(SCHEDULERS)} schedulers × {len(MANTISSA_BITS)} mantissa)")

    count = len(done)
    for method_name, method_kwargs in METHODS:
        for mbits in MANTISSA_BITS:
            label = "fp32" if mbits == 0 else f"m{mbits}"
            for epochs in EPOCHS_LIST:
                for scheduler in SCHEDULERS:
                    for lr in LRS:
                        key = (method_name, lr, epochs, scheduler, mbits)
                        if key in done:
                            continue

                        count += 1
                        r = run_single(method_name, method_kwargs, lr, epochs, scheduler, mbits)
                        results.append(r)

                        if count % 10 == 0:
                            print(f"[{count}/{total}] {method_name} {label} lr={lr:.1e} "
                                  f"ep={epochs} {scheduler[:6]}: "
                                  f"parity={r['parity_acc']:.2%} KL={r['kl']:.4f} "
                                  f"fashion={r['fashion_acc']:.2%}")

                            # Save periodically
                            with open(output_path, "w") as f:
                                json.dump(results, f, indent=2)

    # Final save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone! {len(results)} results saved to {output_path}")


if __name__ == "__main__":
    main()
