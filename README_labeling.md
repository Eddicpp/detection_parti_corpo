# Labeling immagini per YOLO (testa, mano, busto)

## Struttura

- `DATASET/images`: metti qui le immagini da etichettare.
- `DATASET/labels`: verranno creati qui i file YOLO `.txt`.
- `label_yolo.py`: script di labeling manuale.

## Installazione

```bash
pip install opencv-python
```

## Avvio

```bash
python label_yolo.py --dataset-dir DATASET
```

## Comandi durante il labeling

- `1`: classe `testa`
- `2`: classe `mano`
- `3`: classe `busto`
- mouse sinistro + drag: disegna bounding box
- `d`: elimina ultima box
- `c`: cancella tutte le box dell'immagine corrente
- `s`: salva etichette immagine corrente
- `n`: salva e vai alla prossima immagine
- `p`: salva e vai all'immagine precedente
- `q`: salva e chiudi

## Formato output

Per ogni immagine `nome.jpg` viene creato `DATASET/labels/nome.txt` nel formato YOLO:

```text
<class_id> <x_center> <y_center> <width> <height>
```

con coordinate normalizzate in `[0, 1]`.
