#!/usr/bin/env python3
"""GRPO 200 epochs with periodic KL/accuracy eval and model saving."""
import os, sys, json, copy, torch, torch.nn as nn, torch.multiprocessing as mp
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rl_razor.model import MLP
from rl_razor.data import get_finetuning_data, get_fashion_mnist, get_parity_mnist, create_dataloader
from rl_razor.metrics import forward_kl, parity_accuracy
from rl_razor.utils import set_seed

SEED = 42; BS = 64; LR = 1e-4; N_EPOCHS = 200; GROUP_SIZE = 8
EVAL_EVERY = 5
PRETRAIN = "experiments/mantissa_sweep/pretrain/pretrained_model.pt"
OUTPUT_DIR = Path("/mnt/nvme3n1/workspace/osilkin/rl-razor-mnist/experiments/grpo_200ep_eval")

def snap_rne(t, mantissa_bits):
    shift = 23 - mantissa_bits
    halfway = 1 << (shift - 1)
    mask = ~((1 << shift) - 1)
    lsb = 1 << shift
    raw = t.view(torch.int32)
    truncated = raw & mask
    remainder = raw & ((1 << shift) - 1)
    round_up = (remainder > halfway) | ((remainder == halfway) & (((truncated >> shift) & 1) == 1))
    t.copy_((truncated + round_up.int() * lsb).view(torch.float32))

def snap_model(model, mbits):
    if mbits <= 0 or mbits >= 23: return
    with torch.no_grad():
        for p in model.parameters():
            snap_rne(p.data, mbits)

@torch.no_grad()
def evaluate(model, base_model, parity_loader, fashion_loader, device):
    model.eval()
    # Parity accuracy
    c = t = 0
    for x, y in parity_loader:
        x, y = x.to(device), y.to(device)
        c += ((model(x).argmax(1) % 2) == (y % 2)).sum().item(); t += y.size(0)
    par = c / t
    # Fashion accuracy
    c = t = 0
    for x, y in fashion_loader:
        x, y = x.to(device), y.to(device)
        c += (model(x).argmax(1) == y).sum().item(); t += y.size(0)
    fash = c / t
    # Forward KL
    kl = forward_kl(base_model, model, parity_loader, device)
    model.train()
    return par, fash, kl

def worker(gpu_id, mbits):
    device = f'cuda:{gpu_id}'
    set_seed(SEED)
    label = "fp32" if mbits == 23 else f"m{mbits}"

    base = MLP.from_checkpoint(PRETRAIN, device=device)
    model = MLP.from_checkpoint(PRETRAIN, device=device)
    snap_model(model, mbits)
    base.eval()

    train_data, _ = get_finetuning_data("rl")
    loader = create_dataloader(train_data, batch_size=BS, shuffle=True)
    parity_test = get_parity_mnist(train=False)
    parity_loader = create_dataloader(parity_test, batch_size=512, shuffle=False)
    fashion_test = get_fashion_mnist(train=False)
    fashion_loader = create_dataloader(fashion_test, batch_size=512, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    metrics = []

    # Eval epoch 0
    par, fash, kl = evaluate(model, base, parity_loader, fashion_loader, device)
    metrics.append({"epoch": 0, "parity_acc": par, "fashion_acc": fash, "forward_kl": kl})
    print(f"[{label:>4}] ep=0 par={par:.4f} fash={fash:.4f} kl={kl:.4f}", flush=True)

    for ep in range(1, N_EPOCHS + 1):
        model.train()
        epoch_gnorm = 0
        n_steps = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            n = x.size(0)
            opt.zero_grad()

            x_g = x.unsqueeze(1).expand(-1, GROUP_SIZE, -1).reshape(n * GROUP_SIZE, -1)
            y_g = y.unsqueeze(1).expand(-1, GROUP_SIZE).reshape(n * GROUP_SIZE)
            logits = model(x_g)
            probs = torch.softmax(logits, 1)
            log_probs = torch.log_softmax(logits, 1)
            actions = torch.multinomial(probs, 1).squeeze(1)
            rewards = ((actions % 2) == (y_g % 2)).float()
            rewards_g = rewards.reshape(n, GROUP_SIZE)
            advantages = (rewards_g - rewards_g.mean(1, keepdim=True)) / (rewards_g.std(1, keepdim=True) + 1e-8)
            sel_lp = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            loss = -(advantages.reshape(n * GROUP_SIZE) * sel_lp).mean()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')).item()
            epoch_gnorm += gn; n_steps += 1
            opt.step()
            snap_model(model, mbits)

        if ep % EVAL_EVERY == 0 or ep == N_EPOCHS:
            par, fash, kl = evaluate(model, base, parity_loader, fashion_loader, device)
            avg_gn = epoch_gnorm / n_steps
            metrics.append({"epoch": ep, "parity_acc": par, "fashion_acc": fash,
                            "forward_kl": kl, "avg_grad_norm": avg_gn})
            print(f"[{label:>4}] ep={ep} par={par:.4f} fash={fash:.4f} kl={kl:.4f} gnorm={avg_gn:.4f}", flush=True)

    # Save
    run_dir = OUTPUT_DIR / f"grpo_{label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(str(run_dir / "final_model.pt"))
    with open(run_dir / "metrics.json", 'w') as f:
        json.dump({"mantissa_bits": mbits, "metrics": metrics}, f, indent=2)
    print(f"[{label:>4}] Done -> {run_dir}", flush=True)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    procs = []
    for i, mbits in enumerate([7, 8, 9, 10, 23]):
        p = mp.Process(target=worker, args=(i, mbits))
        p.start(); procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            print(f"WARNING: {p.name} exit {p.exitcode}")
    print("Done!")
