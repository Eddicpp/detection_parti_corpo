#!/usr/bin/env python3
"""Utilities for preparing YOLO datasets before training.

Purpose:
- Normalize dataset structure, create train/val split, and generate `data.yaml`.

Datasets expected:
- YOLO datasets with image/label pairs, either flat or already split layout.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import List, Tuple

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = ["testa", "mano", "busto"]


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(
        [
            p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def has_split_structure(dataset_dir: Path) -> bool:
    return (dataset_dir / "images" / "train").exists() and (dataset_dir / "images" / "val").exists()


def _copy_pair(image_path: Path, src_labels: Path, dst_images: Path, dst_labels: Path) -> None:
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    target_image = dst_images / image_path.name
    shutil.copy2(image_path, target_image)

    label_src = src_labels / f"{image_path.stem}.txt"
    label_dst = dst_labels / f"{image_path.stem}.txt"
    if label_src.exists():
        shutil.copy2(label_src, label_dst)
    else:
        label_dst.write_text("", encoding="utf-8")


def _write_data_yaml(base_path: Path, output_path: Path) -> Path:
    data = {
        "path": str(base_path.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": dict(enumerate(CLASS_NAMES)),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return output_path


def _collect_pairs_flat(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    images = list_images(images_dir)
    return [(img, labels_dir / f"{img.stem}.txt") for img in images]


def _collect_pairs_from_split(dataset_dir: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    split_roots = [("train", "train"), ("val", "val")]
    for image_split, label_split in split_roots:
        img_dir = dataset_dir / "images" / image_split
        lbl_dir = dataset_dir / "labels" / label_split
        if not img_dir.exists():
            continue
        for img in list_images(img_dir):
            pairs.append((img, lbl_dir / f"{img.stem}.txt"))
    return pairs


def _copy_to_split(image_path: Path, label_path: Path, out_root: Path, split: str) -> None:
    dst_img_dir = out_root / "images" / split
    dst_lbl_dir = out_root / "labels" / split
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(image_path, dst_img_dir / image_path.name)
    dst_label = dst_lbl_dir / f"{image_path.stem}.txt"
    if label_path.exists():
        shutil.copy2(label_path, dst_label)
    else:
        dst_label.write_text("", encoding="utf-8")


def _build_prepared_from_pairs(
    pairs: List[Tuple[Path, Path]],
    prepared_root: Path,
    val_fraction: float,
    seed: int,
) -> Path:
    if prepared_root.exists():
        shutil.rmtree(prepared_root)

    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)

    if len(shuffled) <= 1:
        val_count = 0
    else:
        val_count = max(1, int(len(shuffled) * val_fraction))

    val_set = set(shuffled[:val_count])
    for image_path, label_path in shuffled:
        split = "val" if (image_path, label_path) in val_set else "train"
        _copy_to_split(image_path, label_path, prepared_root, split)

    return _write_data_yaml(prepared_root, prepared_root / "data.yaml")


def prepare_dataset(
    dataset_dir: Path,
    prepared_root: Path,
    val_fraction: float = 0.2,
    seed: int = 42,
    max_images: int | None = None,
) -> Path:
    """
    Returns a data.yaml ready for YOLO training.

    If dataset already has images/train and images/val, it generates a resolved data yaml.
    If dataset only has images/ and labels/, it creates a train/val split in prepared_root.
    """
    dataset_dir = dataset_dir.resolve()

    if max_images is not None and max_images <= 0:
        raise ValueError("max_images deve essere > 0 oppure None.")

    if has_split_structure(dataset_dir) and max_images is None:
        return _write_data_yaml(dataset_dir, dataset_dir / "data_resolved.yaml")

    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction deve essere compreso tra 0 e 1.")

    if has_split_structure(dataset_dir):
        pairs = _collect_pairs_from_split(dataset_dir)
    else:
        flat_images_dir = dataset_dir / "images"
        flat_labels_dir = dataset_dir / "labels"
        if not flat_labels_dir.exists():
            raise FileNotFoundError(
                f"Cartella labels non trovata: {flat_labels_dir}. Esegui labeling prima del training."
            )
        pairs = _collect_pairs_flat(flat_images_dir, flat_labels_dir)

    if not pairs:
        raise FileNotFoundError(
            f"Nessuna immagine trovata in {dataset_dir}. Aggiungi immagini prima del training."
        )

    rng = random.Random(seed)
    shuffled_pairs = pairs[:]
    rng.shuffle(shuffled_pairs)

    if max_images is not None:
        shuffled_pairs = shuffled_pairs[:max_images]

    return _build_prepared_from_pairs(shuffled_pairs, prepared_root, val_fraction, seed)
