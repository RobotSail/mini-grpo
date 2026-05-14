"""Oracle distribution computation for KL-minimal SFT."""

import torch
import torch.nn.functional as F
from typing import List


# Define which digits have even/odd parity
EVEN_DIGITS = [0, 2, 4, 6, 8]
ODD_DIGITS = [1, 3, 5, 7, 9]


def get_correct_parity_set(label: int) -> List[int]:
    """Get the set of digits with correct parity for a given label.

    Args:
        label: The true digit label (0-9)

    Returns:
        List of digits with the same parity as the label
    """
    if label % 2 == 0:
        return EVEN_DIGITS
    else:
        return ODD_DIGITS


def compute_oracle_distribution(
    base_model: torch.nn.Module,
    x: torch.Tensor,
    label: int,
) -> torch.Tensor:
    """Compute the KL-minimal oracle distribution for a single input.

    The oracle distribution q* minimizes KL(q ∥ π₀) subject to
    q assigning all mass to correct parity labels:

        q* = argmin_q KL(q ∥ π₀)  s.t.  Σ_{y∈S} q(y) = 1

    Note: the paper (Appendix B.3) writes KL(π₀ ∥ q), but that direction
    is undefined (= ∞) when q is constrained to S and π₀ has mass outside S.
    The correct direction is KL(q ∥ π₀), which is the I-projection of π₀
    onto the constraint set (equivalent to rejection sampling; see Lemma A.1).

    Closed-form solution:
        q*(y) = π₀(y) / Σ_{y'∈S} π₀(y') for y ∈ S
        q*(y) = 0 for y ∉ S

    where S is the set of correct parity labels.

    Args:
        base_model: The pretrained base model
        x: Input tensor of shape (1, 785)
        label: True digit label (0-9)

    Returns:
        Oracle distribution tensor of shape (10,)
    """
    base_model.eval()
    with torch.no_grad():
        logits = base_model(x)
        base_probs = F.softmax(logits, dim=-1).squeeze(0)  # Shape: (10,)

    # Get correct parity set
    correct_set = get_correct_parity_set(label)

    # Create mask for correct parity digits
    mask = torch.zeros(10, device=base_probs.device)
    for digit in correct_set:
        mask[digit] = 1.0

    # Compute oracle distribution
    masked_probs = base_probs * mask
    normalizer = masked_probs.sum()

    if normalizer > 0:
        oracle_dist = masked_probs / normalizer
    else:
        # Fallback: uniform over correct set
        oracle_dist = mask / mask.sum()

    return oracle_dist


def compute_oracle_labels_batch(
    base_model: torch.nn.Module,
    x: torch.Tensor,
    labels: torch.Tensor,
    sample: bool = True,
) -> torch.Tensor:
    """Compute oracle labels for a batch of inputs.

    Args:
        base_model: The pretrained base model
        x: Input tensor of shape (batch_size, 785)
        labels: True digit labels of shape (batch_size,)
        sample: If True, sample from oracle distribution; else return argmax

    Returns:
        Oracle labels tensor of shape (batch_size,)
    """
    batch_size = x.size(0)
    device = x.device

    base_model.eval()
    with torch.no_grad():
        logits = base_model(x)
        base_probs = F.softmax(logits, dim=-1)  # Shape: (batch_size, 10)

    # Create masks for correct parity
    even_mask = torch.zeros(10, device=device)
    for d in EVEN_DIGITS:
        even_mask[d] = 1.0

    odd_mask = torch.zeros(10, device=device)
    for d in ODD_DIGITS:
        odd_mask[d] = 1.0

    # Determine which samples have even/odd labels
    is_even = (labels % 2 == 0).float().unsqueeze(1)  # Shape: (batch_size, 1)

    # Create per-sample masks
    masks = is_even * even_mask.unsqueeze(0) + (1 - is_even) * odd_mask.unsqueeze(0)

    # Compute oracle distributions
    masked_probs = base_probs * masks
    normalizers = masked_probs.sum(dim=1, keepdim=True)
    oracle_dists = masked_probs / normalizers.clamp(min=1e-10)

    if sample:
        # Sample from oracle distributions
        oracle_labels = torch.multinomial(oracle_dists, num_samples=1).squeeze(-1)
    else:
        # Take argmax
        oracle_labels = oracle_dists.argmax(dim=-1)

    return oracle_labels


def compute_oracle_loss(
    base_model: torch.nn.Module,
    finetuned_model: torch.nn.Module,
    x: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute cross-entropy loss against oracle distribution.

    Instead of using hard labels, this computes KL divergence from
    the oracle distribution (soft targets).

    Args:
        base_model: The pretrained base model
        finetuned_model: The model being fine-tuned
        x: Input tensor of shape (batch_size, 785)
        labels: True digit labels of shape (batch_size,)

    Returns:
        Scalar loss tensor
    """
    batch_size = x.size(0)
    device = x.device

    # Get base model probabilities
    base_model.eval()
    with torch.no_grad():
        base_logits = base_model(x)
        base_probs = F.softmax(base_logits, dim=-1)

    # Create masks for correct parity
    even_mask = torch.zeros(10, device=device)
    for d in EVEN_DIGITS:
        even_mask[d] = 1.0

    odd_mask = torch.zeros(10, device=device)
    for d in ODD_DIGITS:
        odd_mask[d] = 1.0

    # Determine which samples have even/odd labels
    is_even = (labels % 2 == 0).float().unsqueeze(1)
    masks = is_even * even_mask.unsqueeze(0) + (1 - is_even) * odd_mask.unsqueeze(0)

    # Compute oracle distributions (soft targets)
    masked_probs = base_probs * masks
    normalizers = masked_probs.sum(dim=1, keepdim=True)
    oracle_dists = masked_probs / normalizers.clamp(min=1e-10)

    # Get fine-tuned model log probabilities
    ft_logits = finetuned_model(x)
    ft_log_probs = F.log_softmax(ft_logits, dim=-1)

    # Cross-entropy with soft targets: -Σ q(y) log p(y)
    loss = -(oracle_dists * ft_log_probs).sum(dim=-1).mean()

    return loss
