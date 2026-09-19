import math
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image


@torch.no_grad()
def predict_multiscale(model, image, scales=(0.75, 1.0), flip=False):
    """Predict without image-level labels; output probabilities at original size."""
    if image.ndim == 3:
        image = image.unsqueeze(0)
    shape = image.shape[-2:]
    total = None
    count = 0
    for scale in scales:
        if float(scale) <= 0:
            raise ValueError("Inference scales must be positive")
        size = tuple(max(16, int(math.ceil(v * float(scale) / 16)) * 16) for v in shape)
        resized = F.interpolate(image, size, mode="bilinear", align_corners=False)
        for mirror in ([False, True] if flip else [False]):
            inputs = resized.flip(-1) if mirror else resized
            logits = model(inputs)["seg"]
            if mirror:
                logits = logits.flip(-1)
            probabilities = F.interpolate(logits, shape, mode="bilinear", align_corners=False).softmax(1)
            total = probabilities if total is None else total + probabilities
            count += 1
    if count == 0:
        raise ValueError("At least one scale is required")
    return total / count


def update_histogram(hist, prediction, target, ignore_index=255):
    classes = hist.shape[0]
    valid = (target != ignore_index) & (target >= 0) & (target < classes)
    values = classes * target[valid].astype(np.int64) + prediction[valid].astype(np.int64)
    hist += np.bincount(values, minlength=classes * classes).reshape(classes, classes)


def segmentation_metrics(hist):
    intersection = np.diag(hist)
    union = hist.sum(0) + hist.sum(1) - intersection
    iou = np.divide(intersection, union, out=np.full_like(union, np.nan, dtype=float), where=union > 0)
    mean = float(np.nanmean(iou)) if (union > 0).any() else None
    return {"mIoU": mean, "per_class_IoU": [float(x) if np.isfinite(x) else None for x in iou]}


def save_mask(mask, path):
    """Palette colors are display-only; PNG pixel values remain class IDs."""
    palette = []
    for index in range(256):
        red = green = blue = 0
        value = index
        for shift in range(8):
            red |= (value & 1) << (7 - shift)
            green |= ((value >> 1) & 1) << (7 - shift)
            blue |= ((value >> 2) & 1) << (7 - shift)
            value >>= 3
        palette.extend([red, green, blue])
    image = Image.fromarray(mask.astype(np.uint8), mode="P")
    image.putpalette(palette)
    image.save(path)
