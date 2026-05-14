import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict


def parity_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Compute parity accuracy.

    Parity accuracy measures whether the model correctly predicts
    even vs odd, regardless of the specific digit.

    Args:
        predictions: Predicted digit labels (batch_size,)
        labels: True digit labels (batch_size,)

    Returns:
        Parity accuracy as a float in [0, 1]
    """
    correct = (predictions % 2) == (labels % 2)
    return correct.float().mean().item()


def accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Compute standard classification accuracy.

    Args:
        predictions: Predicted labels (batch_size,)
        labels: True labels (batch_size,)

    Returns:
        Accuracy as a float in [0, 1]
    """
    correct = predictions == labels
    return correct.float().mean().item()


def forward_kl(
    base_model: torch.nn.Module,
    finetuned_model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    eps: float = 1e-10,
) -> float:
    """Compute forward KL divergence from base model to fine-tuned model.

    KL(π₀ || π) = Σ π₀(y|x) log(π₀(y|x) / π(y|x))

    This measures how much the fine-tuned model has diverged from the base model.
    Higher values indicate more divergence (and typically more forgetting).

    Args:
        base_model: The pretrained base model π₀
        finetuned_model: The fine-tuned model π
        dataloader: DataLoader for computing KL over inputs
        device: Device
        eps: Small value for numerical stability

    Returns:
        Average forward KL divergence
    """
    base_model.eval()
    finetuned_model.eval()

    total_kl = 0.0
    total_samples = 0

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            batch_size = x.size(0)

            # Get probabilities from both models
            base_logits = base_model(x)
            ft_logits = finetuned_model(x)

            base_probs = F.softmax(base_logits, dim=-1)
            ft_probs = F.softmax(ft_logits, dim=-1)

            # Clamp for numerical stability
            base_probs = base_probs.clamp(min=eps)
            ft_probs = ft_probs.clamp(min=eps)

            # KL(π₀ || π) = Σ π₀(y) [log π₀(y) - log π(y)]
            kl = (base_probs * (base_probs.log() - ft_probs.log())).sum(dim=-1)

            total_kl += kl.sum().item()
            total_samples += batch_size

    return total_kl / total_samples


def reverse_kl(
    base_model: torch.nn.Module,
    finetuned_model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    eps: float = 1e-10,
) -> float:
    """Compute reverse KL divergence from fine-tuned model to base model.

    KL(π || π₀) = Σ π(y|x) log(π(y|x) / π₀(y|x))

    This is the KL typically used in KL-regularized RL.

    Args:
        base_model: The pretrained base model π₀
        finetuned_model: The fine-tuned model π
        dataloader: DataLoader for computing KL over inputs
        device: Device
        eps: Small value for numerical stability

    Returns:
        Average reverse KL divergence
    """
    base_model.eval()
    finetuned_model.eval()

    total_kl = 0.0
    total_samples = 0

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            batch_size = x.size(0)

            # Get probabilities from both models
            base_logits = base_model(x)
            ft_logits = finetuned_model(x)

            base_probs = F.softmax(base_logits, dim=-1)
            ft_probs = F.softmax(ft_logits, dim=-1)

            # Clamp for numerical stability
            base_probs = base_probs.clamp(min=eps)
            ft_probs = ft_probs.clamp(min=eps)

            # KL(π || π₀) = Σ π(y) [log π(y) - log π₀(y)]
            kl = (ft_probs * (ft_probs.log() - base_probs.log())).sum(dim=-1)

            total_kl += kl.sum().item()
            total_samples += batch_size

    return total_kl / total_samples


def evaluate_model(
    model: torch.nn.Module,
    parity_loader: DataLoader,
    fashion_loader: Optional[DataLoader],
    base_model: Optional[torch.nn.Module],
    device: str,
) -> dict:
    """Comprehensive model evaluation.

    Args:
        model: Model to evaluate
        parity_loader: ParityMNIST dataloader
        fashion_loader: FashionMNIST dataloader (optional)
        base_model: Base model for KL computation (optional)
        device: Device

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    results = {}

    # ParityMNIST parity accuracy
    total_parity_correct = 0
    total_parity_samples = 0

    with torch.no_grad():
        for x, y in parity_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=-1)
            total_parity_correct += ((preds % 2) == (y % 2)).sum().item()
            total_parity_samples += x.size(0)

    results["parity_accuracy"] = total_parity_correct / total_parity_samples

    # FashionMNIST accuracy (measures forgetting)
    if fashion_loader is not None:
        total_fashion_correct = 0
        total_fashion_samples = 0

        with torch.no_grad():
            for x, y in fashion_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=-1)
                total_fashion_correct += (preds == y).sum().item()
                total_fashion_samples += x.size(0)

        results["fashion_accuracy"] = total_fashion_correct / total_fashion_samples

    # KL divergence from base model
    if base_model is not None:
        results["forward_kl"] = forward_kl(base_model, model, parity_loader, device)
        results["reverse_kl"] = reverse_kl(base_model, model, parity_loader, device)

    return results


# ---------------------------------------------------------------------------
# Alternative hypothesis metrics (Section 6 of the paper)
# ---------------------------------------------------------------------------

# ── Distributional distances ────────────────────────────────────────────────


def total_variation(
    base_model: torch.nn.Module,
    finetuned_model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
) -> float:
    """Total variation distance between output distributions, averaged over inputs.

    TV(π₀, π) = 0.5 * E_x[ Σ_y |π₀(y|x) - π(y|x)| ]

    Args:
        base_model: Pretrained base model π₀
        finetuned_model: Fine-tuned model π
        dataloader: DataLoader
        device: Device

    Returns:
        Average total variation distance
    """
    base_model.eval()
    finetuned_model.eval()

    total_tv = 0.0
    total_samples = 0

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            base_probs = F.softmax(base_model(x), dim=-1)
            ft_probs = F.softmax(finetuned_model(x), dim=-1)
            tv = 0.5 * (base_probs - ft_probs).abs().sum(dim=-1)
            total_tv += tv.sum().item()
            total_samples += x.size(0)

    return total_tv / total_samples


def distribution_l2(
    base_model: torch.nn.Module,
    finetuned_model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
) -> float:
    """L2 distance between output distributions, averaged over inputs.

    E_x[ ||π₀(·|x) - π(·|x)||₂ ]

    Args:
        base_model: Pretrained base model π₀
        finetuned_model: Fine-tuned model π
        dataloader: DataLoader
        device: Device

    Returns:
        Average L2 distance between output distributions
    """
    base_model.eval()
    finetuned_model.eval()

    total_dist = 0.0
    total_samples = 0

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            base_probs = F.softmax(base_model(x), dim=-1)
            ft_probs = F.softmax(finetuned_model(x), dim=-1)
            l2 = (base_probs - ft_probs).pow(2).sum(dim=-1).sqrt()
            total_dist += l2.sum().item()
            total_samples += x.size(0)

    return total_dist / total_samples


# ── Weight-level changes ────────────────────────────────────────────────────


def weight_l1(
    base_model: torch.nn.Module,
    finetuned_model: torch.nn.Module,
) -> float:
    """Mean L1 norm of parameter changes (no data required).

    (1/d) * Σᵢ |θᵢ - θ₀ᵢ|

    Args:
        base_model: Pretrained base model
        finetuned_model: Fine-tuned model

    Returns:
        Mean L1 parameter change
    """
    total_l1 = 0.0
    total_params = 0

    base_params = dict(base_model.named_parameters())
    for name, param in finetuned_model.named_parameters():
        delta = param.data - base_params[name].data
        total_l1 += delta.abs().sum().item()
        total_params += param.numel()

    return total_l1 / total_params


def compute_diagonal_fisher(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    n_samples: int = 500,
) -> Dict[str, torch.Tensor]:
    """Compute diagonal Fisher Information Matrix.

    Approximated as the expected squared gradient of the log-likelihood,
    where labels are sampled from the model's own output distribution:

        F_i ≈ (1/N) Σₙ E_{y~π(y|xₙ)} [(∂ log π(y|xₙ) / ∂θᵢ)²]

    Uses per-sample gradients for correctness (batch gradient squaring
    would give (Σ gᵢ)² ≠ Σ gᵢ²).

    Args:
        model: Model to compute Fisher for (use the base model)
        dataloader: DataLoader
        device: Device
        n_samples: Maximum number of samples to use

    Returns:
        Dict mapping parameter name → diagonal Fisher estimate tensor
    """
    model.eval()
    fisher: Dict[str, torch.Tensor] = {name: torch.zeros_like(param.data) for name, param in model.named_parameters()}
    total = 0

    for x, _ in dataloader:
        if total >= n_samples:
            break
        x = x.to(device)
        batch = min(x.size(0), n_samples - total)

        for xi in x[:batch]:
            xi = xi.unsqueeze(0)

            model.zero_grad()
            logits = model(xi)
            probs = F.softmax(logits.detach(), dim=-1)
            y = torch.multinomial(probs, 1).squeeze()
            log_prob = F.log_softmax(logits, dim=-1)[0, y]
            log_prob.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.pow(2)

            total += 1

    for name in fisher:
        fisher[name] /= total

    model.zero_grad()
    return fisher


def weight_fisher_l2(
    base_model: torch.nn.Module,
    finetuned_model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    n_fisher_samples: int = 500,
) -> float:
    """Fisher-weighted L2 distance between parameters (EWC metric).

    Σᵢ F_i * (θᵢ - θ₀ᵢ)²

    where F is the diagonal Fisher of the base model on the given data,
    measuring which parameters mattered most for that data distribution.

    Args:
        base_model: Pretrained base model
        finetuned_model: Fine-tuned model
        dataloader: DataLoader (old or new task)
        device: Device
        n_fisher_samples: Number of samples for Fisher estimation

    Returns:
        Fisher-weighted L2 parameter distance
    """
    fisher = compute_diagonal_fisher(base_model, dataloader, device, n_fisher_samples)

    total = 0.0
    base_params = dict(base_model.named_parameters())
    for name, param in finetuned_model.named_parameters():
        delta = param.data - base_params[name].data
        total += (fisher[name] * delta.pow(2)).sum().item()

    return total


def weight_spectral_norm(
    base_model: torch.nn.Module,
    finetuned_model: torch.nn.Module,
) -> float:
    """Sum of spectral norms of weight-change matrices (no data required).

    For each 2-D weight matrix W: accumulate σ_max(W_ft - W_base).
    Bias vectors are skipped.

    Args:
        base_model: Pretrained base model
        finetuned_model: Fine-tuned model

    Returns:
        Sum of per-layer spectral norms of weight changes
    """
    total_spec = 0.0
    base_params = dict(base_model.named_parameters())

    for name, param in finetuned_model.named_parameters():
        if param.dim() < 2:
            continue  # skip biases
        delta = param.data - base_params[name].data
        # Reshape to 2-D (handles conv layers if ever used)
        delta_2d = delta.view(delta.size(0), -1).float()
        # Largest singular value via SVD
        spec = torch.linalg.matrix_norm(delta_2d, ord=2)
        total_spec += spec.item()

    return total_spec


# ── Activation / representation changes ─────────────────────────────────────


def _get_hidden_activations(
    model: torch.nn.Module,
    x: torch.Tensor,
) -> list:
    """Extract outputs of all ReLU layers via forward hooks.

    Args:
        model: MLP model
        x: Input tensor

    Returns:
        List of activation tensors (one per ReLU layer), each (batch, hidden)
    """
    activations = []

    def _hook(_module, _input, output):
        activations.append(output.detach())

    hooks = [layer.register_forward_hook(_hook) for layer in model.modules() if isinstance(layer, torch.nn.ReLU)]

    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()

    return activations


def activation_distance(
    base_model: torch.nn.Module,
    finetuned_model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    norm: str = "l2",
) -> float:
    """Average hidden-activation distance between base and fine-tuned models.

    Computes the per-sample distance between intermediate representations,
    then averages over ReLU layers and over all samples.

    Args:
        base_model: Pretrained base model
        finetuned_model: Fine-tuned model
        dataloader: DataLoader (old or new task)
        device: Device
        norm: "l1" or "l2"

    Returns:
        Mean activation distance
    """
    base_model.eval()
    finetuned_model.eval()

    total_dist = 0.0
    total_samples = 0

    for x, _ in dataloader:
        x = x.to(device)

        base_acts = _get_hidden_activations(base_model, x)
        ft_acts = _get_hidden_activations(finetuned_model, x)

        layer_dists = []
        for h0, h in zip(base_acts, ft_acts):
            delta = h0 - h
            if norm == "l1":
                d = delta.abs().mean(dim=-1)  # mean over features → (batch,)
            else:
                d = delta.pow(2).mean(dim=-1).sqrt()
            layer_dists.append(d)

        # Average over layers, then sum over batch
        avg = torch.stack(layer_dists, dim=0).mean(dim=0)
        total_dist += avg.sum().item()
        total_samples += x.size(0)

    return total_dist / total_samples


# ── Convenience: compute all Section 6 metrics at once ──────────────────────


def compute_all_alternative_metrics(
    base_model: torch.nn.Module,
    finetuned_model: torch.nn.Module,
    new_task_loader: DataLoader,
    old_task_loader: DataLoader,
    device: str,
    n_fisher_samples: int = 500,
) -> Dict[str, float]:
    """Compute every alternative metric from Table 1 (Section 6).

    Data-dependent metrics are computed on BOTH new-task (ParityMNIST) and
    old-task (FashionMNIST) data, using ``_new`` and ``_old`` suffixes.
    Data-independent weight metrics have no suffix.

    Args:
        base_model: Pretrained base model π₀
        finetuned_model: Fine-tuned model π
        new_task_loader: DataLoader for new task (ParityMNIST)
        old_task_loader: DataLoader for old task (FashionMNIST)
        device: Device
        n_fisher_samples: Samples to use for Fisher estimation

    Returns:
        Dict with keys:
            forward_kl_{new,old}, reverse_kl_{new,old},
            total_variation_{new,old}, distribution_l2_{new,old},
            weight_l1, weight_spectral_norm,
            weight_fisher_l2_{new,old},
            activation_l1_{new,old}, activation_l2_{new,old}
    """
    results: Dict[str, float] = {}

    loaders = [("new", new_task_loader), ("old", old_task_loader)]

    # Distributional distances (both tasks)
    for tag, loader in loaders:
        results[f"forward_kl_{tag}"] = forward_kl(base_model, finetuned_model, loader, device)
        results[f"reverse_kl_{tag}"] = reverse_kl(base_model, finetuned_model, loader, device)
        results[f"total_variation_{tag}"] = total_variation(base_model, finetuned_model, loader, device)
        results[f"distribution_l2_{tag}"] = distribution_l2(base_model, finetuned_model, loader, device)

    # Weight-level changes (no data for L1/spectral; both tasks for Fisher)
    results["weight_l1"] = weight_l1(base_model, finetuned_model)
    results["weight_spectral_norm"] = weight_spectral_norm(base_model, finetuned_model)
    for tag, loader in loaders:
        results[f"weight_fisher_l2_{tag}"] = weight_fisher_l2(
            base_model, finetuned_model, loader, device, n_fisher_samples
        )

    # Activation distances (both tasks)
    for tag, loader in loaders:
        results[f"activation_l1_{tag}"] = activation_distance(base_model, finetuned_model, loader, device, norm="l1")
        results[f"activation_l2_{tag}"] = activation_distance(base_model, finetuned_model, loader, device, norm="l2")

    return results


# ---------------------------------------------------------------------------
# CKNNA — Centered Kernel Alignment with k-NN masking (Huh et al., 2024)
# ---------------------------------------------------------------------------


def _mutual_knn_mask(K_c: torch.Tensor, k: int) -> torch.Tensor:
    """Return a boolean matrix where (i,j) = True iff i and j are mutual
    k-nearest neighbors according to the centered kernel K_c.

    Mutuality means j ∈ kNN(i) **and** i ∈ kNN(j), where neighbors are the
    k entries with the highest kernel value (excluding self-similarity).

    Args:
        K_c: (n, n) centered kernel matrix
        k: number of nearest neighbors

    Returns:
        (n, n) boolean tensor, symmetric, zero diagonal
    """
    n = K_c.shape[0]
    k = min(k, n - 1)
    K_no_diag = K_c.clone()
    K_no_diag.fill_diagonal_(float("-inf"))
    # Top-k indices per row
    _, topk_idx = K_no_diag.topk(k, dim=1)  # (n, k)
    adj = torch.zeros(n, n, dtype=torch.bool, device=K_c.device)
    adj.scatter_(1, topk_idx, True)
    # Mutual: i→j AND j→i
    return adj & adj.T


def compute_cknna(
    X: torch.Tensor,
    Y: torch.Tensor,
    k: int = 10,
) -> float:
    """Centered Kernel Alignment with k-NN masking (CKNNA).

    From Huh et al. 2024, as used in the RL's Razor paper (Appendix B.4):

        K = XX⊤,   L = YY⊤
        K̄ = HKH,   L̄ = HLH    (H = I − (1/n)11⊤)
        α(i,j) = 1 iff i,j are mutual k-NNs in K̄ OR in L̄ (union)
        CKNNA = ⟨K̄,L̄⟩_α / √(⟨K̄,K̄⟩_α · ⟨L̄,L̄⟩_α)

    Setting k = n−1 (all neighbors) recovers standard CKA.

    Args:
        X: (n, d) representation matrix from model A
        Y: (n, d) representation matrix from model B
        k: neighborhood size

    Returns:
        CKNNA score in [−1, 1]; 1.0 = identical local structure
    """
    n = X.shape[0]
    device = X.device

    X = X.float()
    Y = Y.float()

    K = X @ X.T  # (n, n)
    L = Y @ Y.T

    H = torch.eye(n, device=device) - 1.0 / n
    K_c = H @ K @ H
    L_c = H @ L @ H

    mask_K = _mutual_knn_mask(K_c, k)
    mask_L = _mutual_knn_mask(L_c, k)
    alpha = (mask_K | mask_L).float()  # union: mutual k-NNs in either space

    num = (alpha * K_c * L_c).sum()
    dkk = (alpha * K_c * K_c).sum()
    dll = (alpha * L_c * L_c).sum()
    denom = torch.sqrt(dkk * dll)

    if denom.abs() < 1e-12:
        return 1.0  # degenerate: treat as perfectly aligned

    return (num / denom).clamp(-1.0, 1.0).item()


def get_layer_representations(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    layer_idx: int = -1,
    n_samples: Optional[int] = None,
) -> torch.Tensor:
    """Extract hidden-layer activations (after ReLU) for all inputs.

    Args:
        model: MLP model
        dataloader: DataLoader
        device: Device
        layer_idx: Which ReLU layer to extract (−1 = last, 0 = first)
        n_samples: If set, stop after this many samples

    Returns:
        (N, hidden_dim) tensor on CPU
    """
    model.eval()
    all_reprs: list[torch.Tensor] = []
    collected = 0

    for x, _ in dataloader:
        if n_samples is not None and collected >= n_samples:
            break
        x = x.to(device)
        if n_samples is not None:
            x = x[: n_samples - collected]

        acts = _get_hidden_activations(model, x)
        all_reprs.append(acts[layer_idx].cpu())
        collected += x.size(0)

    return torch.cat(all_reprs, dim=0)


def compute_cknna_from_models(
    base_model: torch.nn.Module,
    finetuned_model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    k: int = 10,
    layer_idx: int = -1,
    n_samples: Optional[int] = 2000,
) -> float:
    """Compute CKNNA between base and fine-tuned model representations.

    Args:
        base_model: Pretrained base model π₀
        finetuned_model: Fine-tuned model π
        dataloader: DataLoader for probe inputs
        device: Device
        k: k-NN neighborhood size
        layer_idx: Hidden layer to compare (−1 = last)
        n_samples: Subsample probe data for tractability

    Returns:
        CKNNA score; 1.0 = identical local structure, lower = more drift
    """
    X = get_layer_representations(base_model, dataloader, device, layer_idx, n_samples)
    Y = get_layer_representations(finetuned_model, dataloader, device, layer_idx, n_samples)
    return compute_cknna(X.to(device), Y.to(device), k=k)


def compute_forgetting(
    pretrain_accuracy: float,
    finetune_accuracy: float,
) -> float:
    """Compute forgetting metric.

    Forgetting = (pretrain_accuracy - finetune_accuracy) / pretrain_accuracy

    Args:
        pretrain_accuracy: Accuracy before fine-tuning
        finetune_accuracy: Accuracy after fine-tuning

    Returns:
        Forgetting ratio (0 = no forgetting, 1 = complete forgetting)
    """
    if pretrain_accuracy == 0:
        return 0.0
    return (pretrain_accuracy - finetune_accuracy) / pretrain_accuracy
