# Training YOLO: Locale CUDA + Kaggle

## Prerequisiti locali (RTX 3060)

Installa PyTorch con CUDA e Ultralytics:

```bash
pip install --upgrade pip
pip install ultralytics pyyaml
```

> Se `torch.cuda.is_available()` risulta `False`, reinstalla PyTorch scegliendo la build CUDA dal sito ufficiale PyTorch.

## Script 1: Training locale con CUDA

File: `train_local_cuda.py`

Esempio:

```bash
python train_local_cuda.py \
  --dataset-dir dataset \
  --model yolov8n.pt \
  --epochs 120 \
  --imgsz 640 \
  --batch 16 \
  --device 0
```

Output pesi migliori:
- `runs/body_parts/yolo_cuda/weights/best.pt`

## Script 2: Training su Kaggle GPU

File: `train_kaggle.py`

1. Carica il tuo dataset su Kaggle come Dataset (con cartella `dataset`).
2. Apri un Notebook Kaggle e abilita `Accelerator: GPU`.
3. Nella prima cella installa dipendenze:

```python
!pip install -q ultralytics pyyaml
```

4. Avvia training:

```python
!python train_kaggle.py \
  --dataset-dir /kaggle/input/<nome-dataset>/dataset \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --device 0
```

Output:
- `best.pt` in `/kaggle/working/runs/body_parts/yolo_kaggle/weights/`
- zip artifacts in `/kaggle/working/yolo_kaggle_artifacts.zip`

## Nota su train/val

Gli script gestiscono entrambi i casi:
- Dataset già splittato (`images/train`, `images/val`, `labels/train`, `labels/val`)
- Dataset non splittato (`images`, `labels`)

Se non c'è split, viene creato automaticamente uno split train/val (default 80/20).
