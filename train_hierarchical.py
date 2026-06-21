#!/usr/bin/env python3
"""Train hierarchical YOLO11 models in sequence (person -> body parts).

Purpose:
- Stage 1 trains a person detector on `person_stage/data.yaml`.
- Stage 2 trains a body-parts detector on `parts_stage/data.yaml`.

Datasets expected:
- `person_stage`: YOLO dataset with full images and one class `person`.
- `parts_stage`: YOLO dataset with pre-cropped person images and body-part labels.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hierarchical YOLO11 models (stage1/stage2)")
    parser.add_argument(
        "--dataset-dir-stage1",
        type=Path,
        default=Path("dataset/standardized_datasets/hierarchical/person_stage"),
        help="Stage 1 dataset root containing data.yaml",
    )
    parser.add_argument(
        "--dataset-dir-stage2",
        type=Path,
        default=Path("dataset/standardized_datasets/hierarchical/parts_stage"),
        help="Stage 2 dataset root containing data.yaml",
    )
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Base model")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, 0, 1...")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Validation split fraction (kept for CLI compatibility; datasets are already split)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--project", type=Path, default=Path("runs"), help="Output folder")
    parser.add_argument("--name", type=str, default="hierarchical", help="Run name")
    parser.add_argument(
        "--stage",
        type=str,
        default="both",
        choices=["1", "2", "both"],
        help="Which stage to train",
    )
    return parser.parse_args()


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    return device


def _check_device_or_raise(resolved_device: str) -> None:
    if resolved_device == "cpu":
        print("Device selezionato: CPU")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponibile ma hai richiesto GPU. Usa --device cpu oppure --device auto.")

    device_index = int(resolved_device)
    gpu_name = torch.cuda.get_device_name(device_index)
    print(f"GPU rilevata: {gpu_name}")


def _read_final_metrics(results_csv: Path) -> Dict[str, float | None]:
    metrics = {
        "mAP50": None,
        "mAP50_95": None,
        "precision": None,
        "recall": None,
    }

    if not results_csv.exists():
        return metrics

    with results_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return metrics

    last = rows[-1]

    def pick(keys: list[str]) -> float | None:
        for key in keys:
            val = last.get(key)
            if val is None or val == "":
                continue
            try:
                return float(val)
            except Exception:
                continue
        return None

    metrics["mAP50"] = pick(["metrics/mAP50(B)", "metrics/mAP50"])
    metrics["mAP50_95"] = pick(["metrics/mAP50-95(B)", "metrics/mAP50-95"])
    metrics["precision"] = pick(["metrics/precision(B)", "metrics/precision"])
    metrics["recall"] = pick(["metrics/recall(B)", "metrics/recall"])
    return metrics


def _print_metrics(stage_name: str, metrics: Dict[str, float | None]) -> None:
    print(f"Metriche finali {stage_name}:")
    print(f"  mAP50:      {metrics['mAP50']}")
    print(f"  mAP50-95:   {metrics['mAP50_95']}")
    print(f"  precision:  {metrics['precision']}")
    print(f"  recall:     {metrics['recall']}")


def train_stage1(args: argparse.Namespace, resolved_device: str) -> Dict[str, object]:
    data_yaml = args.dataset_dir_stage1 / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Stage 1 data.yaml non trovato: {data_yaml}")

    run_project = args.project / "stage1_person"
    run_name = args.name

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=resolved_device,
        project=str(run_project),
        name=run_name,
        seed=args.seed,
        pretrained=True,
        cache=True,
    )

    run_dir = run_project / run_name
    results_csv = run_dir / "results.csv"
    best_weights = run_dir / "weights" / "best.pt"
    metrics = _read_final_metrics(results_csv)
    _print_metrics("Stage 1 (Person)", metrics)

    return {
        "ok": True,
        "run_dir": str(run_dir),
        "results_csv": str(results_csv),
        "best_weights": str(best_weights),
        "metrics": metrics,
    }


def train_stage2(args: argparse.Namespace, resolved_device: str) -> Dict[str, object]:
    data_yaml = args.dataset_dir_stage2 / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Stage 2 data.yaml non trovato: {data_yaml}")

    run_project = args.project / "stage2_parts"
    run_name = args.name

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=resolved_device,
        project=str(run_project),
        name=run_name,
        seed=args.seed,
        pretrained=True,
        cache=True,
    )

    run_dir = run_project / run_name
    results_csv = run_dir / "results.csv"
    best_weights = run_dir / "weights" / "best.pt"
    metrics = _read_final_metrics(results_csv)
    _print_metrics("Stage 2 (Body Parts)", metrics)

    return {
        "ok": True,
        "run_dir": str(run_dir),
        "results_csv": str(results_csv),
        "best_weights": str(best_weights),
        "metrics": metrics,
    }


def _save_summary(summary: Dict[str, object], summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> int:
    args = parse_args()

    # Kept for CLI compatibility with previous scripts.
    _ = args.val_fraction

    resolved_device = _resolve_device(args.device)
    _check_device_or_raise(resolved_device)

    summary: Dict[str, object] = {
        "config": {
            "model": args.model,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "device": resolved_device,
            "seed": args.seed,
            "stage": args.stage,
            "dataset_dir_stage1": str(args.dataset_dir_stage1),
            "dataset_dir_stage2": str(args.dataset_dir_stage2),
        },
        "stage1": None,
        "stage2": None,
    }

    summary_path = Path("runs") / "summary.json"

    try:
        if args.stage in ("1", "both"):
            print("[INFO] Avvio Stage 1 - Person Detection")
            summary["stage1"] = train_stage1(args, resolved_device)

        if args.stage in ("2", "both"):
            if args.stage == "both" and not summary.get("stage1"):
                print("[ERROR] Stage 1 non completato: Stage 2 non verrà eseguito.")
                _save_summary(summary, summary_path)
                return 1

            print("[INFO] Avvio Stage 2 - Body Parts Detection")
            summary["stage2"] = train_stage2(args, resolved_device)

    except Exception as exc:
        print(f"[ERROR] Training interrotto: {exc}")
        _save_summary(summary, summary_path)
        return 1

    _save_summary(summary, summary_path)
    print(f"[INFO] Summary salvato in: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
