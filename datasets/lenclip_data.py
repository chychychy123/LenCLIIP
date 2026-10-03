"""Image-level-only training; pixel annotations are opened only for evaluation."""

from pathlib import Path
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class LenCLIPDataset(Dataset):
    def __init__(self, cfg, split, training=False, with_masks=False, with_class_labels=False):
        self.cfg = cfg
        self.name = str(cfg.dataset.name)
        self.root = Path(cfg.dataset.root_dir)
        self.split = str(split)
        self.training = bool(training)
        self.with_masks = bool(with_masks)
        self.with_class_labels = bool(with_class_labels or training)
        if self.training and self.with_masks:
            raise ValueError("Training uses image-level labels only")
        lists = Path(cfg.dataset.name_list_dir)
        self.names = [line.split()[0] for line in (lists / (self.split + ".txt")).read_text().splitlines() if line.strip()]
        if not self.names:
            raise ValueError("The requested split is empty")
        self.labels = None
        if self.with_class_labels:
            # The supplied dictionaries are trusted dataset metadata.
            self.labels = np.load(lists / "cls_labels_onehot.npy", allow_pickle=True).item()
        self.coco_part = "train" if "train" in self.split else "val"

    def __len__(self):
        return len(self.names)

    def image_path(self, name):
        if self.name == "voc":
            return self.root / "JPEGImages" / (name + ".jpg")
        candidates = [
            self.root / "JPEGImages" / self.coco_part / (name + ".jpg"),
            self.root / (self.coco_part + "2014") / (name + ".jpg"),
        ]
        return next((p for p in candidates if p.is_file()), candidates[0])

    def mask_path(self, name):
        if self.name == "voc":
            candidates = [self.root / "SegmentationClass" / (name + ".png"),
                          self.root / "SegmentationClassAug" / (name + ".png")]
        else:
            folder = self.root / "SegmentationClass" / self.coco_part
            number = name.rsplit("_", 1)[-1]
            candidates = [folder / (name + ".png"), folder / (number + ".png")]
            if number.isdigit():
                candidates.append(folder / (str(int(number)) + ".png"))
        return next((p for p in candidates if p.is_file()), candidates[0])

    def augment(self, image):
        scale = random.uniform(*map(float, self.cfg.dataset.rescale_range))
        width, height = image.size
        image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BICUBIC)
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        crop = int(self.cfg.dataset.crop_size)
        width, height = image.size
        canvas = Image.new("RGB", (max(crop, width), max(crop, height)))
        x = random.randint(0, canvas.width - width)
        y = random.randint(0, canvas.height - height)
        canvas.paste(image, (x, y))
        valid = np.zeros((canvas.height, canvas.width), dtype=bool)
        valid[y:y + height, x:x + width] = True
        left = random.randint(0, canvas.width - crop)
        top = random.randint(0, canvas.height - crop)
        return canvas.crop((left, top, left + crop, top + crop)), valid[top:top + crop, left:left + crop].copy()

    def __getitem__(self, index):
        name = self.names[index]
        with Image.open(self.image_path(name)) as source:
            image = source.convert("RGB")
        original_size = (image.height, image.width)
        result = {"name": name, "original_size": original_size}
        if self.training:
            image, valid = self.augment(image)
            result["valid_mask"] = torch.from_numpy(valid)
        array = np.asarray(image, dtype=np.float32) / 255.0
        result["image"] = torch.from_numpy(array.transpose(2, 0, 1).copy())
        if self.with_class_labels:
            if name not in self.labels:
                raise KeyError(f"Missing image-level label for {name}")
            label = np.asarray(self.labels[name], dtype=np.float32).reshape(-1)
            if label.size != int(self.cfg.dataset.num_classes) - 1:
                raise ValueError(f"Wrong class-label dimension for {name}")
            result["class_labels"] = torch.from_numpy((label > 0).astype(np.float32))
        if self.with_masks:
            with Image.open(self.mask_path(name)) as source:
                mask = np.asarray(source, dtype=np.int64).copy()
            if mask.ndim != 2 or mask.shape != original_size:
                raise ValueError(f"Mask for {name} must contain integer class IDs at the original image size")
            allowed = ((mask >= 0) & (mask < int(self.cfg.dataset.num_classes))) | (mask == int(self.cfg.dataset.ignore_index))
            if not allowed.all():
                raise ValueError(f"Mask for {name} contains invalid IDs; remap COCO category IDs to 1..80")
            result["mask"] = torch.from_numpy(mask)
        return result
