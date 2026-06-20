#!/usr/bin/env python3
"""Streamlit UI for YOLO training control and monitoring."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List

import streamlit as st

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency fallback
    pd = None


ROOT_DIR = Path(__file__).resolve().parent
LOCAL_TRAIN_SCRIPT = ROOT_DIR / "train_local_cuda.py"
KAGGLE_TRAIN_SCRIPT = "train_kaggle.py"


def default_run_name() -> str:
    return "yolo_ui_run"


def build_local_command(params: dict) -> List[str]:
    return [
        sys.executable,
        str(LOCAL_TRAIN_SCRIPT),
        "--dataset-dir",
        params["dataset_dir"],
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
        "--max-images",
        str(params["max_images"]),
        "--project",
        params["project"],
        "--name",
        params["name"],
    ]


def build_kaggle_snippet(params: dict) -> str:
    return (
        "!pip install -q ultralytics pyyaml\n"
        "!python "
        f"{KAGGLE_TRAIN_SCRIPT} "
        f"--dataset-dir {params['kaggle_dataset_dir']} "
        f"--model {params['model']} "
        f"--epochs {params['epochs']} "
        f"--imgsz {params['imgsz']} "
        f"--batch {params['batch']} "
        f"--workers {params['workers']} "
        f"--device {params['device']} "
        f"--val-fraction {params['val_fraction']} "
        f"--seed {params['seed']} "
        f"--max-images {params['max_images']} "
        f"--project {params['kaggle_project']} "
        f"--name {params['name']}"
    )


def render_results(project: str, name: str) -> None:
    run_dir = ROOT_DIR / project / name
    results_csv = run_dir / "results.csv"
    best_weights = run_dir / "weights" / "best.pt"

    st.subheader("Risultati")
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


def main() -> None:
    st.set_page_config(page_title="YOLO Trainer UI", layout="wide")
    st.title("YOLO Body Parts Trainer")
    st.caption("Configura parametri, avvia training e monitora i risultati")

    with st.sidebar:
        st.header("Configurazione")
        mode = st.radio("Modalita", ["Locale", "Kaggle"], index=0)

        dataset_dir = st.text_input("Dataset dir (locale)", "dataset/standardized_datasets/unified_body_parts")
        kaggle_dataset_dir = st.text_input("Dataset dir (Kaggle)", "/kaggle/input/body-parts-dataset/dataset")
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
        seed = st.number_input("Seed", min_value=0, max_value=100000, value=42, step=1)
        max_images = st.number_input(
            "Numero immagini da usare",
            min_value=1,
            max_value=1_000_000,
            value=5000,
            step=1,
            help="Limita il dataset per run rapidi. Aumenta per training completo.",
        )

        project = st.text_input("Project output (locale)", "runs/body_parts")
        kaggle_project = st.text_input("Project output (Kaggle)", "/kaggle/working/runs/body_parts")
        name = st.text_input("Run name", default_run_name())

        start = st.button("Avvia training", type="primary")

    params = {
        "dataset_dir": dataset_dir,
        "kaggle_dataset_dir": kaggle_dataset_dir,
        "model": model,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "workers": workers,
        "device": device_choice,
        "val_fraction": round(val_fraction, 4),
        "seed": int(seed),
        "max_images": int(max_images),
        "project": project,
        "kaggle_project": kaggle_project,
        "name": name,
    }

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Parametri correnti")
        st.json(params)

        if mode == "Locale":
            st.info("La modalita locale esegue train_local_cuda.py direttamente da questa UI (CPU/CUDA).")
            if start:
                rc = run_local_training(params)
                if rc == 0:
                    render_results(project, name)
        else:
            st.info("La modalita Kaggle genera il comando pronto da incollare in un notebook Kaggle.")
            snippet = build_kaggle_snippet(params)
            st.code(snippet, language="python")
            if start:
                st.success("Comando Kaggle pronto. Copialo nel notebook Kaggle con GPU attiva.")

    with right:
        st.subheader("Come funziona")
        st.write("1. Imposta i parametri nella sidebar.")
        st.write("2. Avvia training locale oppure genera snippet Kaggle.")
        st.write("3. Segui i log in tempo reale.")
        st.write("4. A fine run, visualizza metriche e percorso best.pt.")


if __name__ == "__main__":
    main()
