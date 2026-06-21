#!/usr/bin/env python3
"""Streamlit UI to train and inspect hierarchical YOLO models.

Purpose:
- Configure and launch hierarchical training (stage 1 person, stage 2 body parts).
- Show run metrics and a visual prediction preview on validation images.

Datasets expected:
- Stage 1 YOLO dataset: person detection in `person_stage` (single class `person`).
- Stage 2 YOLO dataset: body-part detection in `parts_stage` (cropped person images).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List

import cv2
import streamlit as st
from ultralytics import YOLO

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency fallback
    pd = None


ROOT_DIR = Path(__file__).resolve().parent
LOCAL_TRAIN_SCRIPT = ROOT_DIR / "train_hierarchical.py"
KAGGLE_TRAIN_SCRIPT = "train_hierarchical.py"


def default_run_name() -> str:
    return "yolo_ui_run"


def build_local_command(params: dict) -> List[str]:
    return [
        sys.executable,
        str(LOCAL_TRAIN_SCRIPT),
        "--dataset-dir-stage1",
        params["dataset_dir_stage1"],
        "--dataset-dir-stage2",
        params["dataset_dir_stage2"],
        "--model",
        params["model"],
        "--epochs",
        str(params["epochs"]),
        "--imgsz",
        str(params["imgsz"]),
        "--batch",
        str(params["batch"]),
        "--workers",
        str(params["workers"]),
        "--device",
        params["device"],
        "--val-fraction",
        str(params["val_fraction"]),
        "--seed",
        str(params["seed"]),
        "--project",
        params["project"],
        "--name",
        params["name"],
        "--stage",
        params["stage"],
    ]


def build_kaggle_snippet(params: dict) -> str:
    return (
        "!pip install -q ultralytics pyyaml\n"
        "!python "
        f"{KAGGLE_TRAIN_SCRIPT} "
        f"--dataset-dir-stage1 {params['kaggle_dataset_dir_stage1']} "
        f"--dataset-dir-stage2 {params['kaggle_dataset_dir_stage2']} "
        f"--model {params['model']} "
        f"--epochs {params['epochs']} "
        f"--imgsz {params['imgsz']} "
        f"--batch {params['batch']} "
        f"--workers {params['workers']} "
        f"--device {params['device']} "
        f"--val-fraction {params['val_fraction']} "
        f"--seed {params['seed']} "
        f"--project {params['kaggle_project']} "
        f"--name {params['name']} "
        f"--stage {params['stage']}"
    )


def render_results(run_dir: Path, title: str) -> None:
    results_csv = run_dir / "results.csv"
    best_weights = run_dir / "weights" / "best.pt"

    st.subheader(title)
    st.write(f"Run directory: {run_dir}")
    st.write(f"Best weights: {best_weights}")

    if not results_csv.exists():
        st.info("results.csv non trovato: potrebbe non essere stato generato ancora.")
        return

    if pd is None:
        st.info("Installa pandas per vedere i grafici: pip install pandas")
        return

    df = pd.read_csv(results_csv)
    st.dataframe(df.tail(10), use_container_width=True)

    metric_candidates = [
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "train/box_loss",
        "val/box_loss",
    ]

    existing_metrics = [m for m in metric_candidates if m in df.columns]
    if existing_metrics:
        st.line_chart(df[existing_metrics])


def _iter_image_files(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in folder.glob("**/*") if p.is_file() and p.suffix.lower() in exts])


def render_prediction_preview(run_dir: Path, val_images_dir: Path, max_preview_images: int = 10) -> None:
    best_weights = run_dir / "weights" / "best.pt"

    st.subheader("Preview predizioni")

    if not best_weights.exists():
        st.info("best.pt non trovato: impossibile generare preview predizioni.")
        return

    if not val_images_dir.exists():
        st.info("Cartella immagini validation non trovata.")
        return

    image_paths = _iter_image_files(val_images_dir)
    if not image_paths:
        st.info("Nessuna immagine trovata nel validation set preparato.")
        return

    image_paths = image_paths[:max_preview_images]

    try:
        model = YOLO(str(best_weights))
        results = model.predict(
            source=[str(p) for p in image_paths],
            conf=0.1,
            verbose=False,
        )
    except Exception as exc:  # pragma: no cover - runtime dependency path
        st.error(f"Errore durante la preview delle predizioni: {exc}")
        return

    cols = st.columns(2)
    for idx, result in enumerate(results):
        plotted = result.plot()
        rgb_image = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
        with cols[idx % 2]:
            st.image(rgb_image, caption=Path(result.path).name, use_container_width=True)


def run_local_training(params: dict) -> int:
    command = build_local_command(params)

    st.subheader("Comando eseguito")
    st.code(" ".join(command), language="bash")

    log_placeholder = st.empty()
    logs: List[str] = []

    process = subprocess.Popen(
        command,
        cwd=str(ROOT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    for line in process.stdout:
        logs.append(line.rstrip("\n"))
        log_placeholder.code("\n".join(logs[-250:]), language="text")

    return_code = process.wait()

    if return_code == 0:
        st.success("Training completato con successo.")
    else:
        st.error(f"Training terminato con errore (exit code {return_code}).")

    return return_code


def render_locale_mode(params: dict) -> None:
    st.info("La modalita locale esegue train_hierarchical.py direttamente da questa UI (CPU/CUDA).")
    if not params["start"]:
        return

    rc = run_local_training(params)
    if rc != 0:
        return

    project_root = ROOT_DIR / params["project"]
    run_name = params["name"]
    stage = params["stage"]

    if stage in ("1", "both"):
        render_results(project_root / "stage1_person" / run_name, "Risultati Stage 1 (Person)")

    if stage in ("2", "both"):
        stage2_run_dir = project_root / "stage2_parts" / run_name
        render_results(stage2_run_dir, "Risultati Stage 2 (Parts)")
        render_prediction_preview(
            stage2_run_dir,
            ROOT_DIR / params["dataset_dir_stage2"] / "images" / "val",
            max_preview_images=10,
        )


def render_kaggle_mode(params: dict) -> None:
    st.info("La modalita Kaggle genera il comando train_hierarchical.py pronto da incollare in un notebook.")
    snippet = build_kaggle_snippet(params)
    st.code(snippet, language="python")
    if params["start"]:
        st.success("Comando Kaggle pronto. Copialo nel notebook Kaggle con GPU attiva.")


def main() -> None:
    st.set_page_config(page_title="YOLO Trainer UI", layout="wide")
    st.title("YOLO Body Parts Trainer")
    st.caption("Configura parametri, avvia training e monitora i risultati")

    with st.sidebar:
        st.header("Configurazione")
        mode = st.radio("Modalita", ["Locale", "Kaggle"], index=0)

        dataset_dir_stage1 = st.text_input(
            "Dataset stage1 (locale)",
            "dataset/standardized_datasets/hierarchical/person_stage",
        )
        dataset_dir_stage2 = st.text_input(
            "Dataset stage2 (locale)",
            "dataset/standardized_datasets/hierarchical/parts_stage",
        )
        kaggle_dataset_dir_stage1 = st.text_input(
            "Dataset stage1 (Kaggle)",
            "/kaggle/input/hierarchical-dataset/person_stage",
        )
        kaggle_dataset_dir_stage2 = st.text_input(
            "Dataset stage2 (Kaggle)",
            "/kaggle/input/hierarchical-dataset/parts_stage",
        )
        model = st.selectbox(
            "Modello base",
            [
                "yolov8n.pt",
                "yolov8s.pt",
                "yolov8m.pt",
                "yolo11n.pt",
                "yolo11s.pt",
                "yolo11m.pt",
            ],
            index=0,
        )

        epochs = st.slider("Epochs", min_value=1, max_value=500, value=100, step=1)
        imgsz = st.select_slider("Image size", options=[320, 416, 512, 640, 768, 960], value=640)
        batch = st.select_slider("Batch size", options=[4, 8, 16, 24, 32, 48, 64], value=16)
        workers = st.slider("Workers", min_value=0, max_value=16, value=8, step=1)

        if mode == "Locale":
            device_choice = st.selectbox(
                "Device",
                ["auto", "cpu", "0", "1"],
                index=0,
                help="auto: usa GPU se disponibile, altrimenti CPU",
            )
        else:
            device_choice = st.selectbox("Device", ["0", "cpu"], index=0)

        val_fraction = st.slider("Validation fraction", min_value=0.05, max_value=0.5, value=0.2, step=0.01)
        stage = st.selectbox("Stage", ["both", "1", "2"], index=0)
        seed = st.number_input("Seed", min_value=0, max_value=100000, value=42, step=1)

        project = st.text_input("Project output (locale)", "runs")
        kaggle_project = st.text_input("Project output (Kaggle)", "/kaggle/working/runs")
        name = st.text_input("Run name", default_run_name())

        start = st.button("Avvia training", type="primary")

    params = {
        "dataset_dir_stage1": dataset_dir_stage1,
        "dataset_dir_stage2": dataset_dir_stage2,
        "kaggle_dataset_dir_stage1": kaggle_dataset_dir_stage1,
        "kaggle_dataset_dir_stage2": kaggle_dataset_dir_stage2,
        "model": model,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "workers": workers,
        "device": device_choice,
        "val_fraction": round(val_fraction, 4),
        "stage": stage,
        "seed": int(seed),
        "project": project,
        "kaggle_project": kaggle_project,
        "name": name,
        "start": start,
    }

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Parametri correnti")
        st.json(params)

        if mode == "Locale":
            render_locale_mode(params)
        else:
            render_kaggle_mode(params)

    with right:
        st.subheader("Come funziona")
        st.write("1. Imposta i parametri nella sidebar.")
        st.write("2. Avvia training locale oppure genera snippet Kaggle.")
        st.write("3. Segui i log in tempo reale.")
        st.write("4. A fine run, visualizza metriche e percorso best.pt.")


if __name__ == "__main__":
    main()
