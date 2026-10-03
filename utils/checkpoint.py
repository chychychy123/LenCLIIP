from pathlib import Path
import random
import numpy as np
import torch
from omegaconf import OmegaConf


def save_checkpoint(path, model, optimizer, step, cfg, best_miou):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    payload = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "step": int(step), "config": OmegaConf.to_container(cfg, resolve=True),
        "best_miou": float(best_miou), "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, model, optimizer=None, restore_rng=False):
    # Full checkpoints include optimizer/RNG metadata; load only trusted files.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)
    if optimizer is not None:
        if "optimizer" not in checkpoint:
            raise ValueError("Resume requires a full LenCLIP training checkpoint")
        optimizer.load_state_dict(checkpoint["optimizer"])
    if restore_rng:
        torch.set_rng_state(checkpoint["torch_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
        if checkpoint.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    return checkpoint
