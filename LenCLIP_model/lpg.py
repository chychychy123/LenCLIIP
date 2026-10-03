"""Mask generation and image/text matching, with differentiable masked images."""

import math
import torch
from torch import nn
from torch.nn import functional as F


class MaskGenerator(nn.Module):
    def __init__(self, width, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, num_classes, 1),
        )

    def forward(self, features, size):
        logits = self.net(features)
        return F.interpolate(logits, size=size, mode="bilinear", align_corners=False).sigmoid()


def similarity_probability(cosine, mode, eps):
    # The manuscript writes log(cosine), although cosine can be negative.
    # Both the practical shifted domain and literal clamped domain are exposed.
    if mode == "shifted":
        return ((cosine + 1) * 0.5).clamp(eps, 1 - eps)
    if mode == "clamped":
        return cosine.clamp(eps, 1 - eps)
    raise ValueError("loss.cosine_mode must be shifted or clamped")


def region_contrastive_losses(uf, ub, category_ids, texts, options):
    mg, fg, bg = (texts[key][category_ids] for key in ("mg", "foreground", "background"))
    sf = (uf * mg).sum(-1)
    sb = (ub * mg).sum(-1)
    # Exactly Eq. (1): -log(exp(sf)/(exp(sf) + lambda_a * exp(sb))).
    match = F.softplus(sb - sf + math.log(float(options.lambda_a))).mean()
    prob = lambda x: similarity_probability(x, options.cosine_mode, float(options.eps))
    foreground = -prob((uf * fg).sum(-1)).log().mean()
    suppression = -(1 - prob((ub * fg).sum(-1))).log().mean()
    lp_paper = foreground + float(options.lambda_b) * suppression
    # Eqs. (2)-(4) never reference v_b. This explicit, configurable completion
    # supplies the background prompt with gradients; see docs/IMPLEMENTATION.md.
    bg_alignment = -prob((ub * bg).sum(-1)).log().mean()
    bg_suppression = -(1 - prob((uf * bg).sum(-1))).log().mean()
    background_completion = bg_alignment + float(options.lambda_b) * bg_suppression
    lp = lp_paper + float(options.background_prompt_weight) * background_completion
    return {
        "match": match, "foreground": foreground, "background_suppression": suppression,
        "lp_paper": lp_paper, "background_completion": background_completion, "lp": lp,
        "lpg": match + float(options.lambda_lp) * lp,
    }
