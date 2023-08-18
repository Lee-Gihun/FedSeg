import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils import *

__all__ = [
    "evaluate_model_on_loaders",
    "evaluate_model",
]


def sampled_clients_identifier(data_distributed, sampled_clients):
    """Identify local datasets information (distribution, size)"""
    local_dist_list, local_size_list = [], []

    for client_idx in sampled_clients:
        local_dist = torch.Tensor(data_distributed["data_map"])[client_idx]
        local_dist = F.normalize(local_dist, dim=0, p=1)
        local_dist_list.append(local_dist.tolist())

        local_size = data_distributed["local"][client_idx]["datasize"]
        local_size_list.append(local_size)

    return local_dist_list, local_size_list


@torch.no_grad()
def evaluate_model_on_loaders(model, dataloaders, device="cuda:0", prefix="Global"):
    results = {}

    for loader_key, dataloader in dataloaders.items():
        if dataloader is not None:
            key = f"{prefix}_{loader_key}"
            miou, loss = evaluate_model(model, dataloader, device)
            results[key + "_miou"] = miou
            results[key + "_loss"] = loss

    return results


@torch.no_grad()
def evaluate_model(model, dataloader, device="cuda:0"):
    """Evaluate model mIoU for the given dataloader"""

    model.eval()
    model.to(device)

    confusion = np.zeros((19, 19))
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    running_loss, running_counts = 0.0, 0

    for data, targets, _ in dataloader:
        data, targets = data.to(device), targets.to(device)
        running_counts += data.size(0)

        logits = model(data)
        pred = logits.max(dim=1)[1]

        loss = criterion(logits, targets.long())
        running_loss += loss.item() * data.size(0)

        pred = pred.cpu().numpy()
        targets = targets.cpu().numpy()

        for lt, lp in zip(targets, pred):
            confusion += _fast_hist(lt.flatten(), lp.flatten())

    gt_sum = confusion.sum(axis=1)
    mask = gt_sum != 0
    diag = np.diag(confusion)
    iu = diag / (gt_sum + confusion.sum(axis=0) - diag + 1e-6)
    miou = round(np.mean(iu[mask]), 3)
    mloss = round(running_loss / running_counts, 3)

    return miou, mloss


def _fast_hist(label_true, label_pred):
    mask = (label_true >= 0) & (label_true < 19)
    hist = np.bincount(
        19 * label_true[mask].astype(int) + label_pred[mask], minlength=19 ** 2,
    ).reshape(19, 19)
    return hist
