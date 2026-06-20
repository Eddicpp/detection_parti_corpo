#!/usr/bin/env python3
"""Train YOLO on local machine using CUDA or CPU."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

from train_utils import prepare_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO on local machine (CUDA or CPU)")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"), help="Dataset root")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, 0, 1...")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation split fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Numero massimo di immagini da usare (None = tutte)",
    )
    parser.add_argument("--project", type=Path, default=Path("runs/body_parts"), help="Output folder")
    parser.add_argument("--name", type=str, default="yolo_cuda", help="Run name")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.device == "auto":
        resolved_device = "0" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = args.device

    if resolved_device == "cpu":
        print("Device selezionato: CPU")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA non disponibile ma hai richiesto GPU. Usa --device cpu oppure --device auto."
            )
        device_index = int(resolved_device)
        gpu_name = torch.cuda.get_device_name(device_index)
        print(f"GPU rilevata: {gpu_name}")

    prepared_root = args.dataset_dir / "_prepared"
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
        device=resolved_device,
        project=str(args.project),
        name=args.name,
        seed=args.seed,
        pretrained=True,
        cache=True,
    )

    best_weights = args.project / args.name / "weights" / "best.pt"
    print(f"Training completato. Best weights: {best_weights}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
