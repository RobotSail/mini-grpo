import copy
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel


class BertClassifier(nn.Module):

    def __init__(self, model_name: str = "prajjwal1/bert-tiny", num_classes: int = 4):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.bert = AutoModel.from_pretrained(model_name)
        self.head = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.head(out.last_hidden_state[:, 0])

    def copy(self) -> "BertClassifier":
        return copy.deepcopy(self)

    def save_checkpoint(self, path: str, **extra_info):
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": {
                "model_name": self.model_name,
                "num_classes": self.num_classes,
            },
            **extra_info,
        }
        torch.save(checkpoint, path)

    @classmethod
    def from_checkpoint(cls, path: str, device: Optional[str] = None) -> "BertClassifier":
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            config = checkpoint.get("config", {})
        else:
            state_dict = checkpoint
            config = {}

        model = cls(
            model_name=config.get("model_name", "prajjwal1/bert-tiny"),
            num_classes=config.get("num_classes", 4),
        )
        model.load_state_dict(state_dict)
        if device:
            model = model.to(device)
        return model
