#!/usr/bin/env python3
"""
Merge multiple standardized datasets into a single unified dataset.
Useful for combining COCO, custom-labeled, and other datasets.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List

import yaml


def _find_image_label_dirs(dataset_root: Path) -> List[tuple]:
    """Find (images_dir, labels_dir) pairs in dataset."""
    candidates = [
        (dataset_root / "train" / "images", dataset_root / "train" / "labels"),
        (dataset_root / "val" / "images", dataset_root / "val" / "labels"),
        (dataset_root / "images", dataset_root / "labels"),
    ]
    return [(img_d, lbl_d) for img_d, lbl_d in candidates if img_d.exists()]


def _resolve_output_path(img_file: Path, total_images: int, split_ratios: dict, dirs: dict) -> tuple:
    """Determine output (image, label) paths and handle collisions."""
    if total_images % 100 < split_ratios["train"] * 100:
        dst_img_dir = dirs["train_img"]
        dst_lbl_dir = dirs["train_lbl"]
    else:
        dst_img_dir = dirs["val_img"]
        dst_lbl_dir = dirs["val_lbl"]

    dst_img = dst_img_dir / img_file.name
    dst_lbl = dst_lbl_dir / f"{img_file.stem}.txt"

    # Handle name collisions
    counter = 1
    original_stem = img_file.stem
    dataset_name = img_file.parent.parent.parent.name  # dataset root name
    while dst_img.exists():
        new_stem = f"{original_stem}_{dataset_name}_{counter}"
        dst_img = dst_img_dir / f"{new_stem}{img_file.suffix}"
        dst_lbl = dst_lbl_dir / f"{new_stem}.txt"
        counter += 1

    return dst_img, dst_lbl


def _merge_image_label_pairs(
    img_dir: Path,
    label_dir: Path,
    out_dirs: dict,
    split_ratios: dict,
    total_images: int,
) -> int:
    """Merge images and labels from one source directory pair."""
    image_files = sorted(
        [f for f in img_dir.iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    )

    for img_file in image_files:
        label_file = label_dir / f"{img_file.stem}.txt"
        if not label_file.exists():
            print(f"  ⚠ Skipping (no labels): {img_file.name}")
            continue

        dst_img, dst_lbl = _resolve_output_path(img_file, total_images, split_ratios, out_dirs)
        shutil.copy2(img_file, dst_img)
        shutil.copy2(label_file, dst_lbl)
        total_images += 1

        if total_images % 100 == 0:
            print(f"  ✓ Merged {total_images} images...")

    return total_images


def merge_datasets(
    dataset_roots: List[Path],
    output_root: Path,
    split_ratios: dict = None,
) -> None:
    """Merge multiple YOLO-formatted datasets."""
    if split_ratios is None:
        split_ratios = {"train": 0.8, "val": 0.2}

    output_root.mkdir(parents=True, exist_ok=True)
    out_dirs = {
        "train_img": output_root / "images" / "train",
        "train_lbl": output_root / "labels" / "train",
        "val_img": output_root / "images" / "val",
        "val_lbl": output_root / "labels" / "val",
    }
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    total_images = 0

    for dataset_root in dataset_roots:
        print(f"\nMerging from: {dataset_root}")
        dataset_root = dataset_root.resolve()

        for img_dir, label_dir in _find_image_label_dirs(dataset_root):
            total_images = _merge_image_label_pairs(img_dir, label_dir, out_dirs, split_ratios, total_images)

    print(f"\n{'='*60}")
    print(f"Total images merged: {total_images}")
    print(f"Output: {output_root}")
    print(f"{'='*60}")

    data_yaml = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "testa", 1: "mano", 2: "busto"},
    }
    yaml_path = output_root / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, sort_keys=False)
    print(f"Config saved: {yaml_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge multiple standardized datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=Path,
        required=True,
        help="Paths to dataset roots to merge",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/unified_body_parts"),
        help="Output unified dataset path",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training split ratio")
    args = parser.parse_args()

    split_ratios = {"train": args.train_ratio, "val": 1.0 - args.train_ratio}
    merge_datasets(args.datasets, args.output, split_ratios)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
