# YOLO Training Package

Folder ini berisi paket minimal untuk training YOLOv8. Paket ini bisa dipakai di Google Colab atau server lain yang menyediakan Python, CUDA GPU, dan terminal/Jupyter.

## Isi Dataset

```text
data/yolo/images/train: 3050
data/yolo/images/val: 571
data/yolo/images/test: 191
```

Dataset ini sudah berisi tambahan background/negative tile:

```text
train background: 610
val background: 114
test background: 38
total images: 3812
```

`val` tetap dibutuhkan karena YOLO memakainya selama training untuk memilih model terbaik. `test` dipakai setelah training sebagai evaluasi final.

## Setup

```bash
pip install -r requirements-colab.txt
```

Pastikan command dijalankan dari root folder paket ini, yaitu folder yang memiliki `configs`, `data`, `scripts`, dan `src`.

Untuk mengecek GPU NVIDIA:

```bash
nvidia-smi
```

## Training Dengan Background Tiles

```bash
python scripts/train.py \
  --data configs/data.yaml \
  --model yolov8n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --lr0 0.01 \
  --lrf 0.01 \
  --optimizer auto \
  --patience 30 \
  --device 0 \
  --name ship_corrosion_bg_yolov8n
```

Jika GPU kehabisan memory, ganti `--batch 16` menjadi `--batch 8`.

## Evaluasi Test Set

```bash
python scripts/evaluate.py \
  --weights runs/detect/ship_corrosion_bg_yolov8n/weights/best.pt \
  --data configs/data.yaml \
  --imgsz 640 \
  --batch 16 \
  --split test \
  --device 0
```

Metrik yang dicetak:

```text
mAP50-95
mAP50
mAP75
Precision mean
Recall mean
```

## Output Penting

Model terbaik akan berada di:

```text
runs/detect/ship_corrosion_bg_yolov8n/weights/best.pt
```

## Training Dengan Augmentasi Manual

Folder ini juga dapat membawa data augmentasi manual di:

```text
data/processed/augmented/images/train-augment
data/processed/augmented/labels/train-augment
```

Untuk memasukkan data augmentasi manual ke train split:

```bash
python scripts/copy_augmented_to_yolo_train.py --dry-run
python scripts/copy_augmented_to_yolo_train.py
```

Script copy akan mencari folder `train-augment` terlebih dahulu. Jika folder itu tidak ada, script akan fallback ke folder lama bernama `train`.

Setelah disalin, jumlah train menjadi:

```text
3050 current train + 1220 augmented = 4270 train images
```

Training eksperimen augmentasi manual tanpa augmentasi otomatis YOLO:

```bash
python scripts/train.py \
  --data configs/data.yaml \
  --model yolov8n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --lr0 0.01 \
  --lrf 0.01 \
  --optimizer auto \
  --patience 30 \
  --device 0 \
  --name ship_corrosion_manual_aug_no_auto_aug \
  --no-auto-augment
```

Evaluasi test set:

```bash
python scripts/evaluate.py \
  --weights runs/detect/ship_corrosion_manual_aug_no_auto_aug/weights/best.pt \
  --data configs/data.yaml \
  --imgsz 640 \
  --batch 16 \
  --split test \
  --device 0
```
