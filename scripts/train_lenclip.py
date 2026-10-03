"""Train LenCLIP on VOC or COCO with online pseudo labels."""

import argparse
import json
import logging
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from tqdm import tqdm
from datasets import LenCLIPDataset
from LenCLIP_model import LenCLIP
from utils.config import load_config
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.losses import segmentation_loss, affinity_loss
from utils.inference import predict_multiscale, update_histogram, segmentation_metrics


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


@torch.no_grad()
def validate(model, loader, cfg, device):
    model.eval()
    hist = np.zeros((model.num_classes, model.num_classes), dtype=np.int64)
    for batch in tqdm(loader, desc="LenCLIP validation", leave=False):
        probability = predict_multiscale(model, batch["image"].to(device), cfg.inference.scales, cfg.inference.flip)
        prediction = probability.argmax(1)[0].cpu().numpy()
        update_histogram(hist, prediction, batch["mask"][0].numpy(), int(cfg.dataset.ignore_index))
    model.train()
    return segmentation_metrics(hist)


def learning_rate(cfg, step):
    warmup = int(cfg.scheduler.warmup_iters)
    maximum = int(cfg.train.max_iters)
    if warmup > 0 and step < warmup:
        factor = float(cfg.scheduler.warmup_ratio) + (1 - float(cfg.scheduler.warmup_ratio)) * step / warmup
    else:
        progress = (step - warmup) / max(1, maximum - warmup)
        factor = max(0, 1 - progress) ** float(cfg.scheduler.power)
    return float(cfg.optimizer.learning_rate) * factor


def main():
    parser = argparse.ArgumentParser(description="Train LenCLIP using image-level supervision")
    parser.add_argument("--config", default="configs/lenclip_voc.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", help="Full LenCLIP checkpoint")
    parser.add_argument("--work-dir")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    if args.work_dir:
        cfg.train.work_dir = str(Path(args.work_dir).resolve())
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; install a matching PyTorch build or specify --device cpu")
    seed = int(cfg.train.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    work_dir = Path(cfg.train.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[
        logging.StreamHandler(), logging.FileHandler(work_dir / "LenCLIP_train.log", encoding="utf-8"),
    ])
    OmegaConf.save(cfg, work_dir / "LenCLIP_config.yaml")
    training = LenCLIPDataset(cfg, cfg.train.split, training=True)
    workers = int(cfg.train.num_workers)
    loader = DataLoader(training, batch_size=int(cfg.train.batch_size), shuffle=True,
                        num_workers=workers, pin_memory=device.type == "cuda", drop_last=True,
                        worker_init_fn=seed_worker)
    if len(loader) == 0:
        raise ValueError("Training split is smaller than batch_size")
    val_loader = None
    if int(cfg.train.eval_iters) > 0:
        validation = LenCLIPDataset(cfg, cfg.val.split, with_masks=True)
        val_loader = DataLoader(validation, batch_size=1, shuffle=False, num_workers=workers)
    model = LenCLIP(cfg, device=device).to(device).train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(cfg.optimizer.learning_rate),
                                  betas=tuple(cfg.optimizer.betas), weight_decay=float(cfg.optimizer.weight_decay))
    step, best_miou = 0, -1.0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, restore_rng=True)
        step = int(checkpoint["step"])
        best_miou = float(checkpoint.get("best_miou", -1.0))
        logging.info("Resumed LenCLIP at iteration %d", step)
    logging.info("Dataset=%s; trainable parameters=%d; frozen CLIP parameters=%d", cfg.dataset.name,
                 sum(p.numel() for p in parameters), sum(p.numel() for p in model.encoder.parameters()))
    writer = SummaryWriter(str(work_dir / "LenCLIP_tensorboard"))
    iterator = iter(loader)
    start_time = time.time()
    log_values = {}
    log_count = 0
    try:
        while step < int(cfg.train.max_iters):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["class_labels"].to(device, non_blocking=True)
            valid = batch["valid_mask"].to(device, non_blocking=True)
            rate = learning_rate(cfg, step)
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.zero_grad(set_to_none=True)
            output = model(images, labels, step=step, valid_mask=valid)
            seg = segmentation_loss(output["seg"], output["pseudo"], int(cfg.dataset.ignore_index))
            aff = affinity_loss(output["affinity"], output["pseudo"], output["seg"].shape[-2:],
                                int(cfg.loss.affinity_radius), int(cfg.dataset.ignore_index))
            objective = output["losses"]["lpg"]
            total = (float(cfg.loss.seg_weight) * seg + float(cfg.loss.affinity_weight) * aff
                     + float(cfg.loss.lpg_weight) * objective)
            if not torch.isfinite(total):
                raise FloatingPointError(f"Non-finite LenCLIP loss at step {step}; inspect component losses")
            total.backward()
            if float(cfg.train.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(parameters, float(cfg.train.grad_clip))
            optimizer.step()
            step += 1
            values = {key: value.detach().item() for key, value in output["losses"].items()}
            values.update(seg=seg.item(), affinity=aff.item(), total=total.item())
            for key, value in values.items():
                log_values[key] = log_values.get(key, 0.0) + value
            log_count += 1
            if step % int(cfg.train.log_iters) == 0:
                averages = {key: value / log_count for key, value in log_values.items()}
                logging.info("Iter %d/%d lr=%.3g elapsed=%.0fs %s", step, cfg.train.max_iters, rate,
                             time.time() - start_time, " ".join(f"{k}={v:.4f}" for k, v in averages.items()))
                for key, value in averages.items():
                    writer.add_scalar("LenCLIP_loss/" + key, value, step)
                writer.add_scalar("LenCLIP_learning_rate", rate, step)
                log_values, log_count = {}, 0
            if val_loader is not None and step % int(cfg.train.eval_iters) == 0:
                metrics = validate(model, val_loader, cfg, device)
                (work_dir / f"LenCLIP_metrics_{step}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                logging.info("LenCLIP validation: %s", metrics)
                miou = metrics["mIoU"]
                if miou is not None:
                    writer.add_scalar("LenCLIP_val/mIoU", miou, step)
                    if miou > best_miou:
                        best_miou = miou
                        save_checkpoint(work_dir / "checkpoints" / "LenCLIP_best.pth", model, optimizer, step, cfg, best_miou)
            if step % int(cfg.train.save_iters) == 0 or step == int(cfg.train.max_iters):
                path = work_dir / "checkpoints" / f"LenCLIP_iter_{step}.pth"
                save_checkpoint(path, model, optimizer, step, cfg, best_miou)
                logging.info("Saved %s", path)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
