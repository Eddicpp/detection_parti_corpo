#!/usr/bin/env python3
"""Utilities for preparing YOLO datasets for training."""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Iterable, List

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


def prepare_dataset(
    dataset_dir: Path,
    prepared_root: Path,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Path:
    """
    Returns a data.yaml ready for YOLO training.

    If dataset already has images/train and images/val, it generates a resolved data yaml.
    If dataset only has images/ and labels/, it creates a train/val split in prepared_root.
    """
    dataset_dir = dataset_dir.resolve()

    if has_split_structure(dataset_dir):
        return _write_data_yaml(dataset_dir, dataset_dir / "data_resolved.yaml")

    flat_images_dir = dataset_dir / "images"
    flat_labels_dir = dataset_dir / "labels"

    images = list_images(flat_images_dir)
    if not images:
        raise FileNotFoundError(
            f"Nessuna immagine trovata in {flat_images_dir}. Aggiungi immagini prima del training."
        )

    if not flat_labels_dir.exists():
        raise FileNotFoundError(
            f"Cartella labels non trovata: {flat_labels_dir}. Esegui labeling prima del training."
        )

    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction deve essere compreso tra 0 e 1.")

    if prepared_root.exists():
        shutil.rmtree(prepared_root)

    train_images = prepared_root / "images" / "train"
    val_images = prepared_root / "images" / "val"
    train_labels = prepared_root / "labels" / "train"
    val_labels = prepared_root / "labels" / "val"

    rng = random.Random(seed)
    shuffled = images[:]
    rng.shuffle(shuffled)

    val_count = max(1, int(len(shuffled) * val_fraction)) if len(shuffled) > 1 else 0
    val_set = set(shuffled[:val_count])

    for image_path in shuffled:
        if image_path in val_set:
            _copy_pair(image_path, flat_labels_dir, val_images, val_labels)
        else:
            _copy_pair(image_path, flat_labels_dir, train_images, train_labels)

    return _write_data_yaml(prepared_root, prepared_root / "data.yaml")
