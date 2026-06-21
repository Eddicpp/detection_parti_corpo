# Dataset Standardizzazione e Unificazione

Questo modulo standardizza dataset da diversi formati (COCO, custom-labeled, ecc.) in formato YOLO unificato per l'allenamento.

## Architettura

```
dataset/
├── coco2017/                                # Dataset grezzo COCO
│   ├── annotations/
│   │   ├── person_keypoints_train2017.json
│   │   └── person_keypoints_val2017.json
│   ├── train2017/
│   └── val2017/
│
├── standardized_datasets/                   # Cartella di consolidamento (creata automaticamente)
│   ├── coco2017_body_parts/                 # Output di standardize_coco.py
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   ├── val/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   ├── data_train.yaml
│   │   └── data_val.yaml
│   │
│   ├── custom_labeled_body_parts/           # Altro dataset standardizzato (es. da label_yolo.py)
│   │   └── ...
│   │
│   └── unified_body_parts/                  # Output finale di merge_datasets.py
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       ├── labels/
│       │   ├── train/
│       │   └── val/
│       └── data.yaml  <-- Questo file usi per il training YOLO
```

## Step-by-step Workflow

### 1. Standardizzare COCO Dataset

Converte le annotazioni COCO keypoint in bbox YOLO, filtrando per:
- **testa** (keypoint: naso, occhi, orecchi)
- **mano** (keypoint: polsi)
- **busto** (keypoint: spalle, fianchi)

```bash
python standardize_coco.py \
  --coco-root dataset/coco2017 \
  --output-root dataset/standardized_datasets/coco2017_body_parts \
  --split both
```

Output:
- `dataset/standardized_datasets/coco2017_body_parts/train/images` e `labels`
- `dataset/standardized_datasets/coco2017_body_parts/val/images` e `labels`
- File YAML config per ogni split

### 2. (Opzionale) Standardizzare altri dataset

Se hai etichettato immagini con `label_yolo.py`, hanno già il formato YOLO. Puoi metterle in:

```
dataset/
├── custom_labeled/
│   ├── images/
│   └── labels/
```

### 3. Unire tutti i dataset

Combina COCO + custom-labeled + altri in un unico dataset di training:

```bash
python merge_datasets.py \
  --datasets dataset/standardized_datasets/coco2017_body_parts \
             dataset/custom_labeled_body_parts \
  --output dataset/standardized_datasets/unified_body_parts \
  --train-ratio 0.8
```

Output:
- `dataset/standardized_datasets/unified_body_parts/images/train` e `val`
- `dataset/standardized_datasets/unified_body_parts/labels/train` e `val`
- **`dataset/standardized_datasets/unified_body_parts/data.yaml`** ← Usa questo nel training

### 4. Allenare il modello

```bash
python train_hierarchical.py \
  --dataset-dir-stage1 dataset/standardized_datasets/hierarchical/person_stage \
  --dataset-dir-stage2 dataset/standardized_datasets/hierarchical/parts_stage \
  --dataset-dir dataset/standardized_datasets/unified_body_parts \
  --model yolov8n.pt \
  --epochs 100 \
  --batch 16 \
  --device 0
```

## Specifiche Formato YOLO

Per ogni immagine `img.jpg` deve esistere `img.txt` nel formato:

```
<class_id> <x_center> <y_center> <width> <height>
```

Dove:
- `class_id`: 0 (testa), 1 (mano), 2 (busto)
- Coordinate normalizzate in [0, 1]

Esempio:
```
0 0.5234 0.4521 0.2335 0.3121
2 0.6891 0.7234 0.1890 0.2456
1 0.3421 0.2345 0.1234 0.1890
```

## Struttura Directory Finale (Per Training)

Dopo merge, la struttura per training deve essere:

```
unified_body_parts/
├── images/
│   ├── train/   <- 1000+ immagini
│   └── val/     <- 100+ immagini
├── labels/
│   ├── train/   <- File .txt corrispondenti
│   └── val/
└── data.yaml    <- Configurazione YOLO
```

## Note Importanti

1. **Filtraggio automatico**: `standardize_coco.py` estrae SOLO keypoint rilevanti (testa/mano/busto) dalle annotazioni COCO. Ignora tutti gli altri keypoint e label.

2. **Handling collisioni nome**: Se mergi dataset con nomi immagine uguali, `merge_datasets.py` aggiunge suffix automatico.

3. **Dataset bilanciato**: La cartella `unified_body_parts` avrà rapporto train/val configurabile (default 80/20).

4. **YAML config**: Usa sempre il file `data.yaml` finale dalla cartella `unified_body_parts` per training, non dai singoli dataset standardizzati.

## Debugging

Se uno script fallisce:

1. Verifica che i path sorgente esistono:
   ```bash
   ls dataset/coco2017/annotations/person_keypoints_train2017.json
   ```

2. Controlla che le immagini corrispondano alle annotazioni (stesso numero di file).

3. Esamina un file `.txt` di labels per verificare formato:
   ```bash
   head dataset/standardized_datasets/coco2017_body_parts/train/labels/*.txt
   ```
