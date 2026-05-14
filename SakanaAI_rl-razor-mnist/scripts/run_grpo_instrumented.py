#!/usr/bin/env python3
"""GRPO with per-step logging: grad norm, reward, advantage, effective update rate."""
import os, sys, json, copy, torch, torch.nn as nn, torch.multiprocessing as mp
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rl_razor.model import MLP
from rl_razor.data import get_finetuning_data, get_parity_mnist, create_dataloader
from rl_razor.utils import set_seed

SEED = 42; BS = 64; LR = 1e-4; N_EPOCHS = 200; GROUP_SIZE = 8
PRETRAIN = "experiments/mantissa_sweep/pretrain/pretrained_model.pt"
OUTPUT_DIR = Path("experiments/grpo_instrumented_200ep")

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

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    steps_per_epoch = len(loader)

    step_log = []
    global_step = 0

    for ep in range(1, N_EPOCHS + 1):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            n = x.size(0)
            opt.zero_grad()

            # Save pre-step weights
            pre_weights = {name: p.data.clone() for name, p in model.named_parameters()}

            x_g = x.unsqueeze(1).expand(-1, GROUP_SIZE, -1).reshape(n * GROUP_SIZE, -1)
            y_g = y.unsqueeze(1).expand(-1, GROUP_SIZE).reshape(n * GROUP_SIZE)

            logits = model(x_g)
            probs = torch.softmax(logits, 1)
            log_probs = torch.log_softmax(logits, 1)
            actions = torch.multinomial(probs, 1).squeeze(1)

            rewards = ((actions % 2) == (y_g % 2)).float()
            rewards_g = rewards.reshape(n, GROUP_SIZE)
            advantages = (rewards_g - rewards_g.mean(1, keepdim=True)) / (rewards_g.std(1, keepdim=True) + 1e-8)
            advantages = advantages.reshape(n * GROUP_SIZE)

            sel_lp = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            loss = -(advantages * sel_lp).mean()
            loss.backward()

            # Gradient norm
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')).item()

            opt.step()
            snap_model(model, mbits)

            # Effective update: how many params actually changed?
            total_params = 0
            changed_params = 0
            with torch.no_grad():
                for name, p in model.named_parameters():
                    total_params += p.numel()
                    changed_params += (p.data != pre_weights[name]).sum().item()

            global_step += 1
            epoch_frac = global_step / steps_per_epoch

            entry = {
                "step": global_step,
                "epoch": round(epoch_frac, 4),
                "grad_norm": grad_norm,
                "loss": loss.item(),
                "mean_reward": rewards.mean().item(),
                "mean_abs_advantage": advantages.abs().mean().item(),
                "effective_update_rate": changed_params / total_params,
            }
            step_log.append(entry)

            if global_step % 50 == 0 or global_step <= 5:
                print(f"[{label:>4}] step={global_step:>4} ep={epoch_frac:.2f} "
                      f"gnorm={grad_norm:.4f} reward={rewards.mean():.3f} "
                      f"|adv|={advantages.abs().mean():.3f} "
                      f"update_rate={changed_params/total_params:.4f}", flush=True)

    out_path = OUTPUT_DIR / f"grpo_{label}_steps.json"
    with open(out_path, 'w') as f:
        json.dump({"mantissa_bits": mbits, "steps": step_log}, f)
    print(f"[{label:>4}] Done -> {out_path}", flush=True)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    configs = [7, 8, 9, 10, 23]
    procs = []
    for i, mbits in enumerate(configs):
        p = mp.Process(target=worker, args=(i, mbits))
        p.start(); procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            print(f"WARNING: {p.name} exit {p.exitcode}")
    print("Done!")
