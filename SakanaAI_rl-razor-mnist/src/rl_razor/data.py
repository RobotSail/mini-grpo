"""Data loading for ParityMNIST and FashionMNIST experiments."""

import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
from typing import Tuple, Optional, Literal
import numpy as np


class TaskIndicatorDataset(Dataset):
    """Wraps a dataset to add a task indicator to each sample.

    The task indicator is concatenated to the flattened image:
    - +1 for ParityMNIST
    - -1 for FashionMNIST
    """

    def __init__(
        self,
        base_dataset: Dataset,
        task_indicator: float,
        parity_labels: bool = False,
        label_mode: Optional[Literal["sft1", "sft2"]] = None,
    ):
        """
        Args:
            base_dataset: The underlying MNIST or FashionMNIST dataset
            task_indicator: +1 for ParityMNIST, -1 for FashionMNIST
            parity_labels: If True, sample label uniformly from correct-parity digits
                (used during pretraining to provide a valid, but randomly chosen, target)
            label_mode: Label mode for SFT training:
                - "sft1": Even->0, Odd->1
                - "sft2": Even->{0,4} random, Odd->{1,5} random
                - None: return the original MNIST label unchanged
                Note: oracle SFT training is handled externally via
                compute_oracle_loss() and does not use this dataset parameter.
        """
        self.base_dataset = base_dataset
        self.task_indicator = task_indicator
        self.parity_labels = parity_labels
        self.label_mode = label_mode

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, label = self.base_dataset[idx]

        # Flatten image (28x28 = 784)
        x = img.view(-1)

        # Append task indicator
        x = torch.cat([x, torch.tensor([self.task_indicator])])

        # Determine output label
        if self.parity_labels:
            # For pretraining: sample uniformly from correct parity class
            parity = label % 2
            correct_digits = [d for d in range(10) if d % 2 == parity]
            output_label = torch.tensor(np.random.choice(correct_digits))
        elif self.label_mode == "sft1":
            # SFT-1: Even->0, Odd->1
            output_label = torch.tensor(label % 2)
        elif self.label_mode == "sft2":
            # SFT-2: Even->{0,4} random, Odd->{1,5} random
            parity = label % 2
            if parity == 0:
                output_label = torch.tensor(np.random.choice([0, 4]))
            else:
                output_label = torch.tensor(np.random.choice([1, 5]))
        else:
            # Standard label (used for RL reward computation and oracle SFT)
            output_label = torch.tensor(label)

        return x, output_label


def get_mnist_base(train: bool = True, data_dir: str = "./data") -> Dataset:
    """Get base MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return datasets.MNIST(data_dir, train=train, download=True, transform=transform)


def get_fashion_mnist_base(train: bool = True, data_dir: str = "./data") -> Dataset:
    """Get base FashionMNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    return datasets.FashionMNIST(data_dir, train=train, download=True, transform=transform)


def get_parity_mnist(
    train: bool = True,
    data_dir: str = "./data",
    parity_labels: bool = False,
    label_mode: Optional[Literal["sft1", "sft2"]] = None,
) -> Dataset:
    """Get ParityMNIST dataset with task indicator +1.

    Args:
        train: If True, return training set
        data_dir: Directory to store/load data
        parity_labels: If True, sample labels uniformly from correct parity class
            (used for pretraining only; not suitable for deterministic evaluation)
        label_mode: Label mode for SFT ("sft1", "sft2", or None for standard labels)
            Oracle SFT labels are computed on-the-fly via compute_oracle_loss(),
            so "oracle" is not a valid label_mode here.

    Returns:
        Dataset with (x, label) where x has 785 dimensions
    """
    base = get_mnist_base(train=train, data_dir=data_dir)
    return TaskIndicatorDataset(
        base,
        task_indicator=1.0,
        parity_labels=parity_labels,
        label_mode=label_mode,
    )


def get_fashion_mnist(
    train: bool = True,
    data_dir: str = "./data",
) -> Dataset:
    """Get FashionMNIST dataset with task indicator -1.

    Args:
        train: If True, return training set
        data_dir: Directory to store/load data

    Returns:
        Dataset with (x, label) where x has 785 dimensions
    """
    base = get_fashion_mnist_base(train=True if train else False, data_dir=data_dir)
    return TaskIndicatorDataset(base, task_indicator=-1.0)


def get_pretraining_data(
    n_samples_per_task: int = 500,
    data_dir: str = "./data",
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """Get pretraining datasets: 500 samples each from ParityMNIST and FashionMNIST.

    Args:
        n_samples_per_task: Number of samples to use from each task
        data_dir: Directory to store/load data
        seed: Random seed for reproducible subset selection

    Returns:
        Tuple of (combined_train_dataset, combined_val_dataset)
    """
    rng = np.random.RandomState(seed)

    # Get full datasets
    parity_train = get_parity_mnist(train=True, data_dir=data_dir, parity_labels=True)
    fashion_train = get_fashion_mnist(train=True, data_dir=data_dir)

    # Sample indices
    parity_indices = rng.choice(len(parity_train), n_samples_per_task, replace=False)
    fashion_indices = rng.choice(len(fashion_train), n_samples_per_task, replace=False)

    # Create subsets
    parity_subset = Subset(parity_train, parity_indices)
    fashion_subset = Subset(fashion_train, fashion_indices)

    # Combine
    train_dataset = ConcatDataset([parity_subset, fashion_subset])

    # Validation sets (use test sets).
    # parity_labels=False so validation labels are the actual MNIST digits —
    # deterministic and correct for parity-accuracy evaluation.
    parity_val = get_parity_mnist(train=False, data_dir=data_dir, parity_labels=False)
    fashion_val = get_fashion_mnist(train=False, data_dir=data_dir)

    val_dataset = ConcatDataset([parity_val, fashion_val])

    return train_dataset, val_dataset


def get_finetuning_data(
    label_mode: Literal["sft1", "sft2", "rl"] = "rl",
    data_dir: str = "./data",
) -> Tuple[Dataset, Dataset]:
    """Get fine-tuning datasets for ParityMNIST only.

    Args:
        label_mode: How to generate labels
            - "sft1": Even->0, Odd->1
            - "sft2": Even->{0,4}, Odd->{1,5}
            - "rl": Standard MNIST labels (for GRPO reward computation and oracle SFT)
        data_dir: Directory to store/load data

    Returns:
        Tuple of (train_dataset, val_dataset)

    Note:
        Oracle SFT uses label_mode="rl" and computes soft targets on-the-fly via
        compute_oracle_loss() in training/sft.py.
    """
    if label_mode == "rl":
        train = get_parity_mnist(train=True, data_dir=data_dir)
        val = get_parity_mnist(train=False, data_dir=data_dir)
    else:
        train = get_parity_mnist(
            train=True,
            data_dir=data_dir,
            label_mode=label_mode,
        )
        val = get_parity_mnist(
            train=False,
            data_dir=data_dir,
            label_mode=label_mode,
        )

    return train, val


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader with standard settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
