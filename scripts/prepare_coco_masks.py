"""Optional evaluation-mask conversion; never called by LenCLIP training."""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Convert COCO instance annotations to 1..80 semantic IDs")
    parser.add_argument("--annotations", required=True, help="instances_val2014.json")
    parser.add_argument("--output", required=True, help="MSCOCO/SegmentationClass/val")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    from pycocotools.coco import COCO
    coco = COCO(args.annotations)
    category_ids = sorted(coco.getCatIds())
    if len(category_ids) != 80:
        raise ValueError("Expected the 80 COCO instance categories")
    mapping = {category: index + 1 for index, category in enumerate(category_ids)}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for image_id in tqdm(sorted(coco.getImgIds()), desc="COCO semantic masks"):
        image = coco.loadImgs([image_id])[0]
        destination = output / (Path(image["file_name"]).stem + ".png")
        if destination.exists() and not args.overwrite:
            continue
        mask = np.zeros((image["height"], image["width"]), dtype=np.uint8)
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=[image_id]))
        crowd = np.zeros_like(mask, dtype=bool)
        # Larger regions first: smaller instances retain their visible pixels.
        for annotation in sorted(annotations, key=lambda x: x.get("area", 0), reverse=True):
            region = coco.annToMask(annotation).astype(bool)
            if annotation.get("iscrowd", 0):
                crowd |= region
            else:
                mask[region] = mapping[annotation["category_id"]]
        mask[crowd & (mask == 0)] = 255
        Image.fromarray(mask).save(destination)
    print(f"Saved masks to {output.resolve()}")


if __name__ == "__main__":
    main()
