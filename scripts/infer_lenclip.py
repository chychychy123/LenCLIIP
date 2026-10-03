"""LenCLIP multi-scale segmentation on dataset splits, an image, or a folder."""

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import LenCLIPDataset
from LenCLIP_model import LenCLIP
from utils.config import load_config
from utils.checkpoint import load_checkpoint
from utils.inference import predict_multiscale, update_histogram, segmentation_metrics, save_mask


def local_images(path):
    path = Path(path)
    files = sorted(p for p in path.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")) if path.is_dir() else [path]
    if not files:
        raise ValueError("No input images found")
    for file in files:
        with Image.open(file) as source:
            image = np.asarray(source.convert("RGB"), dtype=np.float32) / 255
        yield {"name": [file.name], "image": torch.from_numpy(image.transpose(2, 0, 1).copy()).unsqueeze(0)}


def main():
    parser = argparse.ArgumentParser(description="LenCLIP inference")
    parser.add_argument("--config", default="configs/lenclip_voc.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="LenCLIP_predictions")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", help="Dataset split; defaults to val.split")
    parser.add_argument("--input", help="Single RGB image or image folder")
    parser.add_argument("--evaluate", action="store_true", help="Read pixel masks and compute mIoU")
    parser.add_argument("--scales", nargs="+", type=float)
    parser.add_argument("--flip", action="store_true", default=None)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    if args.input and (args.evaluate or args.split):
        parser.error("--input cannot be combined with --evaluate or --split")
    cfg = load_config(args.config, args.set)
    device = torch.device(args.device)
    model = LenCLIP(cfg, device=device).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()
    scales = args.scales if args.scales else cfg.inference.scales
    flip = bool(cfg.inference.flip) if args.flip is None else args.flip
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.input:
        batches = local_images(args.input)
    else:
        dataset = LenCLIPDataset(cfg, args.split or cfg.val.split, with_masks=args.evaluate)
        batches = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=int(cfg.inference.num_workers))
    histogram = np.zeros((model.num_classes, model.num_classes), dtype=np.int64)
    for batch in tqdm(batches, desc="LenCLIP inference"):
        probabilities = predict_multiscale(model, batch["image"].to(device), scales, flip)
        prediction = probabilities.argmax(1)[0].cpu().numpy()
        name = batch["name"][0]
        # Preserve the original extension for arbitrary files to avoid name clashes.
        filename = name + ".png" if args.input else Path(name).stem + ".png"
        save_mask(prediction, output / filename)
        if args.evaluate:
            update_histogram(histogram, prediction, batch["mask"][0].numpy(), int(cfg.dataset.ignore_index))
    if args.evaluate:
        result = segmentation_metrics(histogram)
        (output / "LenCLIP_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    print(f"LenCLIP predictions saved to {output.resolve()}")


if __name__ == "__main__":
    main()
