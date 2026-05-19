import random

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from datasets import load_dataset

NEG_CLASSES = [0, 1]
POS_CLASSES = [2, 3]


class SentimentDataset(Dataset):
    """Amazon sentiment with on-the-fly label mapping.

    label_mode:
      pretrain / sft2: neg -> random({0,1}), pos -> random({2,3})
      sft1: neg -> 0, pos -> 2
      rl: binary label passed through (for GRPO/oracle reward)
    """

    def __init__(self, input_ids, attention_mask, sentiment_binary, label_mode="pretrain"):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.sentiment_binary = sentiment_binary
        self.label_mode = label_mode

    def __len__(self):
        return len(self.sentiment_binary)

    def __getitem__(self, idx):
        sent = self.sentiment_binary[idx].item()

        if self.label_mode in ("pretrain", "sft2"):
            label = random.choice(NEG_CLASSES) if sent == 0 else random.choice(POS_CLASSES)
        elif self.label_mode == "sft1":
            label = 0 if sent == 0 else 2
        else:
            label = sent

        return (self.input_ids[idx], self.attention_mask[idx],
                torch.tensor(label, dtype=torch.long),
                self.sentiment_binary[idx])


def build_ag_news(texts, labels, tokenizer, max_len, max_samples=None, seed=42):
    idx = list(range(len(texts)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    if max_samples:
        idx = idx[:max_samples]

    sel_texts = ["news: " + texts[i] for i in idx]
    sel_labels = torch.tensor([labels[i] for i in idx], dtype=torch.long)

    enc = tokenizer(sel_texts, truncation=True, padding="max_length",
                    max_length=max_len, return_tensors="pt")
    return TensorDataset(enc["input_ids"], enc["attention_mask"], sel_labels)


def build_sentiment(texts, labels, tokenizer, max_len, label_mode="pretrain",
                    max_samples=None, seed=42):
    idx = list(range(len(texts)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    if max_samples:
        idx = idx[:max_samples]

    sel_texts = ["sentiment: " + texts[i] for i in idx]
    sentiment_binary = torch.tensor([labels[i] for i in idx], dtype=torch.long)

    enc = tokenizer(sel_texts, truncation=True, padding="max_length",
                    max_length=max_len, return_tensors="pt")
    return SentimentDataset(enc["input_ids"], enc["attention_mask"],
                            sentiment_binary, label_mode=label_mode)


def get_pretraining_data(tokenizer, max_len, n_samples_per_task=5000, seed=42):
    """Load and tokenize pretraining data. Returns (ag_dataset, sentiment_dataset)."""
    ag = load_dataset("ag_news")
    ag_ds = build_ag_news(ag["train"]["text"], ag["train"]["label"],
                          tokenizer, max_len, max_samples=n_samples_per_task, seed=seed)

    amz = load_dataset("amazon_polarity", split=f"train[:{n_samples_per_task}]")
    sent_ds = build_sentiment(amz["content"], amz["label"],
                              tokenizer, max_len, label_mode="pretrain")

    return ag_ds, sent_ds


def get_finetuning_data(tokenizer, max_len, label_mode="sft2", n_samples=50000,
                        n_skip=5000, seed=42):
    """Load and tokenize fine-tuning sentiment data (non-overlapping with pretraining)."""
    amz = load_dataset("amazon_polarity",
                        split=f"train[{n_skip}:{n_skip + n_samples}]")
    return build_sentiment(amz["content"], amz["label"],
                           tokenizer, max_len, label_mode=label_mode)


def get_ag_news_val(tokenizer, max_len, max_samples=5000, seed=42):
    ag = load_dataset("ag_news")
    return build_ag_news(ag["test"]["text"], ag["test"]["label"],
                         tokenizer, max_len, max_samples=max_samples, seed=seed)


def get_sentiment_val(tokenizer, max_len, max_samples=5000, seed=42):
    amz = load_dataset("amazon_polarity", split=f"test[:{max_samples}]")
    return build_sentiment(amz["content"], amz["label"],
                           tokenizer, max_len, label_mode="pretrain")
