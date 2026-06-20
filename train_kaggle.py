#!/usr/bin/env python3
"""Train YOLO on Kaggle GPU environment."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

from train_utils import prepare_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO on Kaggle")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/kaggle/input/body-parts-dataset/dataset"),
        help="Path del dataset in Kaggle Input",
    )
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--workers", type=int, default=2, help="Dataloader workers")
    parser.add_argument("--device", type=str, default="0", help="CUDA device id")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation split fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Numero massimo di immagini da usare (None = tutte)",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("/kaggle/working/runs/body_parts"),
        help="Output folder",
    )
    parser.add_argument("--name", type=str, default="yolo_kaggle", help="Run name")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("GPU non disponibile in Kaggle. Abilita Accelerator: GPU nelle impostazioni notebook.")

    prepared_root = Path("/kaggle/working/prepared_dataset")
    data_yaml = prepare_dataset(
        dataset_dir=args.dataset_dir,
        prepared_root=prepared_root,
        val_fraction=args.val_fraction,
        seed=args.seed,
        max_images=args.max_images,
    )
    print(f"Dataset YOLO pronto: {data_yaml}")

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(args.project),
        name=args.name,
        seed=args.seed,
        pretrained=True,
        cache=True,
    )

    run_dir = args.project / args.name
    best_weights = run_dir / "weights" / "best.pt"
    print(f"Training completato. Best weights: {best_weights}")

    archive_base = Path("/kaggle/working") / f"{args.name}_artifacts"
    archive_path = shutil.make_archive(str(archive_base), "zip", str(run_dir))
    print(f"Artifact zip creato: {archive_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
