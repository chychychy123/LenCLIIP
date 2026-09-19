"""Class-aware local feature propagation, Eq. (5), without dense NxN VA maps."""

import torch
from torch import nn
from torch.nn import functional as F


class VisualAlignment(nn.Module):
    def __init__(self, radius=1, iterations=1, visual_temperature=0.1,
                 semantic_temperature=0.1, background_scale=0.2, enabled=True):
        super().__init__()
        self.radius = int(radius)
        self.iterations = int(iterations)
        self.visual_temperature = float(visual_temperature)
        self.semantic_temperature = float(semantic_temperature)
        self.background_scale = float(background_scale)
        self.enabled = bool(enabled)
        if self.radius < 0 or self.iterations < 0:
            raise ValueError("VA radius and iterations must be nonnegative")
        if min(self.visual_temperature, self.semantic_temperature) <= 0:
            raise ValueError("VA temperatures must be positive")

    def forward(self, features, foreground, background, class_labels):
        b, d, h, w = features.shape
        k = foreground.shape[0]
        kernel = 2 * self.radius + 1
        count = kernel * kernel
        base = F.normalize(features, dim=1)
        refined_text = F.normalize(foreground - self.background_scale * background, dim=-1)
        valid = F.unfold(base.new_ones((1, 1, h, w)), kernel, padding=self.radius).bool()
        outputs = []
        # Process each image and its positive classes; absent classes remain zero.
        for batch in range(b):
            image_cams = []
            for category in range(k):
                if not bool(class_labels[batch, category] > 0):
                    image_cams.append(base.new_zeros(h, w))
                    continue
                current = base[batch:batch + 1]
                text = foreground[category].view(1, d, 1, 1)
                if self.enabled:
                    for _ in range(self.iterations):
                        neighbor = F.unfold(current, kernel, padding=self.radius).view(1, d, count, h * w)
                        center = current.flatten(2).unsqueeze(2)
                        visual = (center * neighbor).sum(1) / self.visual_temperature
                        response = (current * text).sum(1, keepdim=True)
                        neighbor_response = F.unfold(response, kernel, padding=self.radius)
                        semantic = -(neighbor_response - response.flatten(2)).abs() / self.semantic_temperature
                        weights = (visual + semantic).masked_fill(~valid, float("-inf")).softmax(1)
                        current = F.normalize((neighbor * weights.unsqueeze(1)).sum(2).view(1, d, h, w), dim=1)
                cam = (current[0] * refined_text[category, :, None, None]).sum(0).relu()
                image_cams.append(cam / cam.amax().clamp_min(1e-6))
            outputs.append(torch.stack(image_cams))
        return torch.stack(outputs)
