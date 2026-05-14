import torch
import torch.nn as nn
from typing import Optional


class MLP(nn.Module):
    """3-layer MLP for ParityMNIST / FashionMNIST experiments.

    Architecture:
        Input: 785 (784 flattened image + 1 task indicator)
        Hidden: 512 -> 256
        Output: 10 (digit classes)
        Activations: ReLU
    """

    def __init__(
        self,
        input_dim: int = 785,
        hidden_dims: tuple = (512, 256),
        output_dim: int = 10,
        dropout: float = 0.0,
    ):
        """
        Args:
            input_dim: Input dimension (784 + 1 task indicator = 785)
            hidden_dims: Tuple of hidden layer dimensions
            output_dim: Number of output classes
            dropout: Dropout probability (0 = no dropout)
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits.

        Args:
            x: Input tensor of shape (batch_size, 785)

        Returns:
            Logits tensor of shape (batch_size, 10)
        """
        return self.network(x)

    def get_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Get softmax probabilities.

        Args:
            x: Input tensor of shape (batch_size, 785)

        Returns:
            Probability tensor of shape (batch_size, 10)
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        """Sample actions from the policy.

        Args:
            x: Input tensor of shape (batch_size, 785)

        Returns:
            Sampled actions of shape (batch_size,)
        """
        probs = self.get_probs(x)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def copy(self) -> "MLP":
        """Create a deep copy of this model."""
        import copy

        return copy.deepcopy(self)

    @classmethod
    def from_checkpoint(cls, path: str, device: Optional[str] = None) -> "MLP":
        """Load model from checkpoint.

        Args:
            path: Path to checkpoint file
            device: Device to load model to

        Returns:
            Loaded MLP model
        """
        checkpoint = torch.load(path, map_location=device)

        # Handle both full checkpoints and state_dict only
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            config = checkpoint.get("config", {})
        else:
            state_dict = checkpoint
            config = {}

        model = cls(
            input_dim=config.get("input_dim", 785),
            hidden_dims=tuple(config.get("hidden_dims", (512, 256))),
            output_dim=config.get("output_dim", 10),
        )
        model.load_state_dict(state_dict)

        if device:
            model = model.to(device)

        return model

    def save_checkpoint(self, path: str, **extra_info):
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
            **extra_info: Additional info to save (e.g., epoch, optimizer state)
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": {
                "input_dim": self.input_dim,
                "hidden_dims": self.hidden_dims,
                "output_dim": self.output_dim,
            },
            **extra_info,
        }
        torch.save(checkpoint, path)
