#!/usr/bin/env python3
"""
Run all 5 methods at m7/bf16 with train/val/test evaluation at every checkpoint.
Uses the SakanaAI GRPO implementation.
"""
import os, sys, json, copy, torch, torch.nn as nn, torch.multiprocessing as mp
from torch.distributions import Categorical
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rl_razor.utils import set_seed

SEED = 42
N_EPOCHS = 100
LR = 1e-4
BS = 64
MANTISSA = 7
CKPT_EVERY = 0.2  # 5 checkpoints per epoch
PRETRAIN_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "mini-grpo", "parity_mnist_v3", "base_model.pt")
OUTPUT_DIR = Path("experiments/bf16_checkpoint_eval")


class MLP(nn.Module):
    """Same architecture as parity_mnist_v3 (784 input, no task indicator)."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 10),
        )
    def forward(self, x):
        return self.layers(x)
    def copy(self):
        m = MLP()
        m.load_state_dict(copy.deepcopy(self.state_dict()))
        return m


def snap_to_lattice(model, mantissa_bits):
    if mantissa_bits <= 0 or mantissa_bits >= 23:
        return
    shift = 23 - mantissa_bits
    halfway = 1 << (shift - 1)
    mask = ~((1 << shift) - 1)
    lsb = 1 << shift
    with torch.no_grad():
        for p in model.parameters():
            raw = p.data.view(torch.int32)
            truncated = raw & mask
            remainder = raw & ((1 << shift) - 1)
            round_up = (remainder > halfway) | (
                (remainder == halfway) & (((truncated >> shift) & 1) == 1))
            p.data.copy_((truncated + round_up.int() * lsb).view(torch.float32))


@torch.no_grad()
def eval_all(model, train_loader, val_loader, test_loader, fashion_loader, base_model, device):
    model.eval()
    def _parity(m, loader):
        c = t = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            c += ((m(x).argmax(1) % 2) == (y % 2)).sum().item(); t += y.size(0)
        return c / t
    def _class(m, loader):
        c = t = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            c += (m(x).argmax(1) == y).sum().item(); t += y.size(0)
        return c / t
    def _kl(base, ft, loader):
        base.eval(); ft.eval()
        kl_sum = n = 0
        for x, _ in loader:
            x = x.to(device)
            lp = torch.log_softmax(base(x), 1)
            lq = torch.log_softmax(ft(x), 1)
            kl_sum += (lp.exp() * (lp - lq)).sum().item(); n += x.size(0)
        return kl_sum / n

    return {
        "train_parity": _parity(model, train_loader),
        "val_parity": _parity(model, val_loader),
        "test_parity": _parity(model, test_loader),
        "fashion_acc": _class(model, fashion_loader),
        "forward_kl": _kl(base_model, model, test_loader),
    }


def worker(gpu_id, method, out_path):
    device = f"cuda:{gpu_id}"
    set_seed(SEED)

    base_model = MLP().to(device)
    base_model.load_state_dict(torch.load(PRETRAIN_PATH, map_location=device, weights_only=True))

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_full = datasets.MNIST("data", True, download=False, transform=tfm)
    mnist_test = datasets.MNIST("data", False, download=False, transform=tfm)
    fmnist_test = datasets.FashionMNIST("data", False, download=False, transform=tfm)

    # 80/20 train/val split
    n_val = int(len(mnist_full) * 0.2)
    n_train = len(mnist_full) - n_val
    train_set, val_set = random_split(mnist_full, [n_train, n_val],
                                       generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_set, BS, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, 512, num_workers=2)
    test_loader = DataLoader(mnist_test, 512, num_workers=2)
    fashion_loader = DataLoader(fmnist_test, 512, num_workers=2)

    model = MLP().to(device)
    model.load_state_dict(copy.deepcopy(base_model.state_dict()))
    snap_to_lattice(model, MANTISSA)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    ce = nn.CrossEntropyLoss()

    steps_per_epoch = len(train_loader)
    total_steps = N_EPOCHS * steps_per_epoch
    # Checkpoint every 0.2 epochs = 5 per epoch
    ckpt_interval = max(1, int(steps_per_epoch * CKPT_EVERY))

    metrics = []
    global_step = 0

    # Record epoch 0
    m = eval_all(model, train_loader, val_loader, test_loader, fashion_loader, base_model, device)
    m["epoch"] = 0.0
    m["step"] = 0
    metrics.append(m)
    print(f"[{method:>8s}] ep=0.0 train={m['train_parity']:.4f} val={m['val_parity']:.4f} "
          f"test={m['test_parity']:.4f} fash={m['fashion_acc']:.4f} kl={m['forward_kl']:.4f}", flush=True)

    for ep in range(1, N_EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()

            if method in ("sft1", "sft2", "oracle"):
                if method == "sft1":
                    targets = (y % 2).long()  # even->0, odd->1
                elif method == "sft2":
                    # even -> {0,4} random, odd -> {1,5} random
                    targets = y.clone()
                    even = y % 2 == 0
                    odd = ~even
                    targets[even] = torch.where(torch.rand(even.sum(), device=device) < 0.5,
                                                 torch.zeros(even.sum(), device=device, dtype=torch.long),
                                                 torch.full((even.sum(),), 4, device=device, dtype=torch.long))
                    targets[odd] = torch.where(torch.rand(odd.sum(), device=device) < 0.5,
                                                torch.ones(odd.sum(), device=device, dtype=torch.long),
                                                torch.full((odd.sum(),), 5, device=device, dtype=torch.long))
                else:  # oracle
                    with torch.no_grad():
                        base_logits = base_model(x)
                        base_probs = torch.softmax(base_logits, dim=-1)
                    # Mask to correct parity, renormalize, sample
                    parity = y % 2
                    mask = torch.zeros(x.size(0), 10, device=device)
                    for i in range(10):
                        mask[:, i] = ((i % 2) == parity).float()
                    oracle_probs = base_probs * mask
                    oracle_probs = oracle_probs / oracle_probs.sum(dim=-1, keepdim=True)
                    targets = torch.multinomial(oracle_probs, 1).squeeze(-1)

                loss = ce(model(x), targets)
            else:  # grpo or grpo_kl
                group_size = 8
                n = x.size(0)
                x_g = x.unsqueeze(1).expand(-1, group_size, -1, -1, -1).reshape(n * group_size, *x.shape[1:])
                y_g = y.unsqueeze(1).expand(-1, group_size).reshape(n * group_size)

                logits = model(x_g)
                probs = torch.softmax(logits, dim=-1)
                log_probs = torch.log_softmax(logits, dim=-1)
                actions = torch.multinomial(probs, 1).squeeze(-1)

                rewards = ((actions % 2) == (y_g % 2)).float()
                rewards_g = rewards.reshape(n, group_size)
                advantages = (rewards_g - rewards_g.mean(dim=1, keepdim=True))
                advantages = advantages / (rewards_g.std(dim=1, keepdim=True) + 1e-8)
                advantages = advantages.reshape(n * group_size)

                sel_lp = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
                pg_loss = -(advantages * sel_lp).mean()

                kl_loss = 0.0
                if method == "grpo_kl":
                    with torch.no_grad():
                        base_lp = torch.log_softmax(base_model(x_g), dim=-1)
                    kl_loss = (probs * (log_probs - base_lp)).sum(dim=-1).mean()

                loss = pg_loss + 0.1 * kl_loss

            loss.backward()
            opt.step()
            snap_to_lattice(model, MANTISSA)
            global_step += 1

            if global_step % ckpt_interval == 0:
                epoch_frac = global_step / steps_per_epoch
                m = eval_all(model, train_loader, val_loader, test_loader, fashion_loader, base_model, device)
                m["epoch"] = round(epoch_frac, 2)
                m["step"] = global_step
                metrics.append(m)

                if int(epoch_frac) == epoch_frac or epoch_frac <= 2:
                    print(f"[{method:>8s}] ep={epoch_frac:.1f} train={m['train_parity']:.4f} "
                          f"val={m['val_parity']:.4f} test={m['test_parity']:.4f} "
                          f"fash={m['fashion_acc']:.4f} kl={m['forward_kl']:.4f}", flush=True)

    with open(out_path, "w") as f:
        json.dump({"method": method, "mantissa_bits": MANTISSA, "metrics": metrics}, f, indent=2)
    print(f"[{method:>8s}] Done -> {out_path}", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    methods = ["grpo", "grpo_kl", "sft1", "sft2", "oracle"]
    procs = []
    for i, method in enumerate(methods):
        out = str(OUTPUT_DIR / f"{method}_m7.json")
        if Path(out).exists():
            print(f"SKIP {method}")
            continue
        p = mp.Process(target=worker, args=(i, method, out))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
        if p.exitcode != 0:
            print(f"WARNING: {p.name} exit {p.exitcode}")

    print("\nDone!")
