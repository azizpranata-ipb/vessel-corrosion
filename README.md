# Ship Hull Corrosion Detection with YOLOv8

Sistem end-to-end untuk mendeteksi korosi lambung kapal dari foto inspeksi drone menggunakan YOLOv8 Detection, lalu melakukan segmentasi area korosi di dalam bounding box dengan OpenCV untuk menghitung luas, Dice Coefficient, dan tingkat keparahan.

## Asumsi

- Dataset memiliki 1 class: `corrosion`.
- Training memakai YOLOv8 object detection, bukan segmentation.
- Label YOLO berupa bounding box. Ground truth mask bersifat opsional dan hanya dipakai jika ingin menghitung Dice Coefficient terhadap mask manual.
- Perhitungan luas default memakai satuan piksel. Jika tersedia faktor kalibrasi `mm_per_pixel`, luas juga dihitung dalam `cm2`.
- Severity memakai aturan awal berbasis rasio area korosi terhadap area gambar dan bisa diubah di `configs/app.yaml`.

## Struktur Project

```text
ship-corrosion-yolo/
├── api/
│   ├── main.py
│   └── static/
│       ├── app.html
│       ├── login.html
│       ├── login.js
│       ├── styles.css
│       └── app.js
├── configs/
│   ├── app.yaml
│   └── data.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── yolo/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── labels/
│           ├── train/
│           ├── val/
│           └── test/
├── models/
├── outputs/
│   ├── predictions/
│   └── reports/
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── infer_image.py
├── src/
│   └── corrosion/
│       ├── __init__.py
│       ├── config.py
│       ├── dice.py
│       ├── inference.py
│       ├── schema.py
│       ├── segmentation.py
│       ├── severity.py
│       └── visualization.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Format Dataset YOLOv8

Setiap gambar memiliki file label `.txt` dengan nama sama:

```text
data/yolo/images/train/ship_001.jpg
data/yolo/labels/train/ship_001.txt
```

Isi label YOLO:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Semua koordinat dinormalisasi 0 sampai 1. Karena hanya 1 class:

```text
0 0.5125 0.4380 0.2200 0.1800
```

Konfigurasi dataset ada di `configs/data.yaml`.

## Instalasi Lokal

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Di Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Training YOLOv8

Letakkan dataset dalam format YOLOv8, lalu jalankan:

```bash
python scripts/train.py --data configs/data.yaml --model yolov8n.pt --epochs 100 --imgsz 640 --batch 16
```

### Cropping/Tiling Gambar Besar

Untuk gambar lambung kapal yang sangat lebar, detail korosi kecil bisa hilang jika seluruh gambar langsung di-resize ke `imgsz 640`. Gunakan tiling agar gambar dipotong menjadi patch `640x640` tanpa mengecilkan detail.

Contoh untuk gambar dan label yang sudah dipisah:

```bash
python scripts/tile_yolo_dataset.py \
  --images data/raw/images \
  --labels data/raw/labels \
  --out-images data/yolo/images/train \
  --out-labels data/yolo/labels/train \
  --tile-size 640 \
  --overlap 128 \
  --keep-empty
```

Jika belum ada label dan hanya ingin memotong gambar:

```bash
python scripts/tile_yolo_dataset.py \
  --images data/raw/images \
  --out-images data/processed/tiles \
  --tile-size 640 \
  --overlap 128 \
  --keep-empty
```

`--overlap 128` membuat antar patch saling tumpang tindih agar korosi di batas tile tidak hilang. Jika label YOLO tersedia, script akan otomatis menyesuaikan koordinat bounding box ke setiap tile.

Hasil training tersimpan di:

```text
runs/detect/ship_corrosion/weights/best.pt
```

## Evaluasi Model

```bash
python scripts/evaluate.py --weights runs/detect/ship_corrosion/weights/best.pt --data configs/data.yaml
```

Evaluasi YOLO menghasilkan precision, recall, mAP50, dan mAP50-95. Dice Coefficient untuk segmentasi OpenCV dapat dihitung jika Anda menyediakan mask ground truth.

## Inferensi Gambar Tunggal

```bash
python scripts/infer_image.py --weights runs/detect/ship_corrosion/weights/best.pt --image path/to/image.jpg
```

Output:

- JSON hasil analisis.
- Gambar anotasi di `outputs/predictions`.

## REST API

Jalankan:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Buka:

```text
http://localhost:8000
```

Login awal:

```text
admin / admin123
user / user123
```

Role `user` hanya bisa upload dan analisis gambar. Role `admin` bisa upload gambar serta membuka menu manajemen user untuk tambah dan hapus user. Data user disimpan di `data/users.json` dan password disimpan sebagai hash PBKDF2.

Endpoint utama:

```text
POST /api/predict
```

Form-data:

- `file`: gambar inspeksi.
- `mm_per_pixel`: opsional, contoh `0.5`.

## Format JSON Response API

```json
{
  "image": {
    "filename": "ship.jpg",
    "width": 1280,
    "height": 720
  },
  "summary": {
    "detections": 2,
    "total_corrosion_area_px": 18340,
    "total_corrosion_area_cm2": null,
    "corrosion_ratio": 0.0199,
    "severity": "moderate"
  },
  "detections": [
    {
      "id": 1,
      "class_name": "corrosion",
      "confidence": 0.86,
      "bbox_xyxy": [120, 80, 360, 240],
      "bbox_area_px": 38400,
      "corrosion_area_px": 9200,
      "corrosion_area_cm2": null,
      "corrosion_ratio_in_bbox": 0.2396,
      "severity": "moderate",
      "dice_coefficient": null
    }
  ],
  "artifacts": {
    "annotated_image_url": "/outputs/predictions/annotated_xxx.jpg",
    "mask_image_url": "/outputs/predictions/mask_xxx.png"
  }
}
```

## Docker

Build dan jalankan:

```bash
docker compose up --build
```

Aplikasi tersedia di:

```text
http://localhost:8000
```

Gunakan volume `./models:/app/models` untuk memasukkan model production, misalnya:

```text
models/best.pt
```

### Docker Production

Copy env template:

```bash
cp .env.example .env
```

Pastikan model sudah tersedia:

```text
models/best.pt
```

Jalankan mode production:

```bash
docker compose -f docker-compose.prod.yml up -d --build
```

Cek container:

```bash
docker compose -f docker-compose.prod.yml ps
docker logs -f corrosion-api
```

Buka aplikasi:

```text
http://SERVER_IP:8000
```

Data yang dipertahankan di luar container:

- `data/users.json`: database user lokal.
- `models/best.pt`: file model YOLOv8.
- `outputs/`: upload dan hasil prediksi.

Jika ingin push image ke Docker Hub atau registry cloud:

```bash
docker build -t username/ship-corrosion-yolo:1.0.0 .
docker push username/ship-corrosion-yolo:1.0.0
```

Di server, ubah `image` pada `docker-compose.prod.yml` menjadi image registry tersebut, lalu jalankan:

```bash
docker compose -f docker-compose.prod.yml pull
docker compose -f docker-compose.prod.yml up -d
```

## Deployment ke VPS Linux

1. Siapkan VPS Ubuntu 22.04/24.04.
2. Install Docker dan Compose plugin.
3. Clone repository ke server.
4. Upload model YOLOv8 terbaik ke `models/best.pt`.
5. Sesuaikan `.env` atau environment variable:

```bash
export MODEL_PATH=models/best.pt
export APP_CONFIG=configs/app.yaml
```

6. Jalankan:

```bash
docker compose up -d --build
```

7. Pasang reverse proxy Nginx:

```nginx
server {
    server_name corrosion.example.com;

    client_max_body_size 25M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

8. Aktifkan HTTPS dengan Certbot.

## Best Practice Production

- Simpan model di volume terpisah, bukan di image Docker.
- Gunakan model path dari environment variable.
- Batasi ukuran upload di FastAPI dan Nginx.
- Validasi MIME type dan ekstensi file.
- Simpan hasil prediksi dengan UUID agar nama file tidak bentrok.
- Pisahkan training environment dan serving environment.
- Jalankan API di belakang Nginx dengan HTTPS.
- Aktifkan logging request dan error.
- Monitor latency, jumlah deteksi, confidence rata-rata, dan ukuran file.
- Versioning model: `models/yolov8_corrosion_v1.pt`, `v2.pt`, dan seterusnya.
- Untuk luas fisik yang valid, lakukan kalibrasi kamera/drone dan simpan `mm_per_pixel` atau homography per inspeksi.
- Untuk Dice Coefficient yang sah secara akademik, siapkan ground truth mask manual atau semi-manual, karena bounding box saja tidak cukup sebagai ground truth segmentasi.
