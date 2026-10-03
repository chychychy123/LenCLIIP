import torch
from torch.nn import functional as F


def segmentation_loss(logits, target, ignore_index=255):
    logits = F.interpolate(logits, target.shape[-2:], mode="bilinear", align_corners=False)
    pixel_loss = F.cross_entropy(logits, target.long(), ignore_index=ignore_index, reduction="none")
    valid = target != ignore_index
    groups = [valid & (target == 0), valid & (target != 0)]
    terms = [pixel_loss[mask].mean() for mask in groups if mask.any()]
    return torch.stack(terms).mean() if terms else logits.sum() * 0


def affinity_loss(affinity, target, grid_size, radius=8, ignore_index=255):
    down = F.interpolate(target[:, None].float(), grid_size, mode="nearest")[:, 0].long().flatten(1)
    valid = down != ignore_index
    pair_valid = valid[:, :, None] & valid[:, None, :]
    y, x = torch.meshgrid(torch.arange(grid_size[0], device=target.device),
                          torch.arange(grid_size[1], device=target.device), indexing="ij")
    coords = torch.stack((y.flatten(), x.flatten()), dim=-1)
    difference = (coords[:, None] - coords[None, :]).abs()
    local = difference.amax(-1) <= radius
    local.fill_diagonal_(False)
    pair_valid = pair_valid & local
    same = down[:, :, None] == down[:, None, :]
    positive, negative = pair_valid & same, pair_valid & ~same
    terms = []
    if positive.any():
        terms.append((1 - affinity[positive]).mean())
    if negative.any():
        terms.append(affinity[negative].mean())
    return torch.stack(terms).mean() if terms else affinity.sum() * 0
