import torch
import torch.nn.functional as F


@torch.no_grad()
def news_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for ids, mask, labels in loader:
        ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
        correct += (model(ids, mask).argmax(1) == labels).sum().item()
        total += labels.size(0)
    return correct / total


@torch.no_grad()
def sentiment_accuracy(model, loader, device):
    """Parity-style accuracy: correct if pred in {0,1} for neg, {2,3} for pos."""
    model.eval()
    correct = total = 0
    for batch in loader:
        ids, mask = batch[0].to(device), batch[1].to(device)
        sentiment_binary = batch[3].to(device)
        preds = model(ids, mask).argmax(1)
        pred_sentiment = (preds >= 2).long()
        correct += (pred_sentiment == sentiment_binary).sum().item()
        total += sentiment_binary.size(0)
    return correct / total


@torch.no_grad()
def forward_kl(ref_model, model, loader, device):
    """KL(p_ref || p_model) over 4-class softmax, averaged per sample."""
    ref_model.eval()
    model.eval()
    kl_sum = 0.0
    n = 0
    for batch in loader:
        ids, mask = batch[0].to(device), batch[1].to(device)
        ref_logp = F.log_softmax(ref_model(ids, mask), dim=1)
        cur_logp = F.log_softmax(model(ids, mask), dim=1)
        kl = (ref_logp.exp() * (ref_logp - cur_logp)).sum(1)
        kl_sum += kl.sum().item()
        n += ids.size(0)
    return kl_sum / n
