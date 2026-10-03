from pathlib import Path
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path, overrides=None):
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    for section, key in (("dataset", "root_dir"), ("dataset", "name_list_dir"),
                         ("model", "clip_pretrain_path"), ("train", "work_dir")):
        value = Path(str(cfg[section][key])).expanduser()
        cfg[section][key] = str(value if value.is_absolute() else PROJECT_ROOT / value)
    if str(cfg.dataset.name) not in ("voc", "coco"):
        raise ValueError("dataset.name must be voc or coco")
    if int(cfg.dataset.crop_size) <= 0 or int(cfg.dataset.crop_size) % 16:
        raise ValueError("dataset.crop_size must be a positive multiple of 16")
    if int(cfg.train.batch_size) < 1 or int(cfg.train.max_iters) < 1:
        raise ValueError("batch_size and max_iters must be positive")
    if int(cfg.train.save_iters) < 1 or int(cfg.train.log_iters) < 1:
        raise ValueError("save_iters and log_iters must be positive")
    if not 0 < float(cfg.pseudo.par_scale) <= 1:
        raise ValueError("pseudo.par_scale must be in (0, 1]")
    if min(float(cfg.loss.lambda_a), float(cfg.loss.eps)) <= 0:
        raise ValueError("lambda_a and eps must be positive")
    return cfg
