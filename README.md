# Concert Surveillance: Violence & Dangerous Object Detection

Training notebooks for the models behind a crowd-surveillance system for large events such as concerts. The project trains two kinds of models:

1. **Action recognition**: a Temporal Shift Module (TSM) on a MobileNetV2 backbone that classifies short video clips as `normal` or `violence`.
2. **Object detection**: a YOLO26n detector for `person`, `knife`, `weapon`, and `hug`.

The models are used by the real-time inference pipeline in [tsm-grid-camera](https://github.com/harrymardika/tsm-grid-camera).

## Results

| Model | Data | Validation result |
|---|---|---|
| TSM, whole-frame clips (`action.ipynb`) | 250 clips (210 normal, 40 violence), 200 train / 50 validation | 92.0% accuracy, recall 1.00, precision 0.67, F1 0.80 (saved checkpoint, epoch 7) |
| TSM, per-person clips (`action_per_person.ipynb`) | 1,327 clips (812 normal, 515 violence), 1,061 train / 266 validation | 89.1% accuracy, precision 0.90, recall 0.82, F1 0.86 (saved checkpoint, epoch 9) |
| YOLO26n detector (`object.ipynb`) | 1,040 validation images, 3,017 instances | mAP50 0.877, mAP50-95 0.636 |

YOLO26n per-class results:

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| hug | 0.937 | 0.961 | 0.985 | 0.762 |
| knife | 0.866 | 0.751 | 0.837 | 0.529 |
| person | 0.888 | 0.872 | 0.920 | 0.726 |
| weapon | 0.880 | 0.649 | 0.765 | 0.527 |

The whole-frame model reaches high accuracy on a small validation set, but its precision on the violence class is low (0.67). Training on per-person clips with a larger dataset gave a more balanced model. Per-epoch metrics are in `models/training_log.csv` and `models_per_person/training_log_per_person.csv`.

## Approach

### Action recognition (TSM)

- **Frame extraction:** videos are split into JPEG frames per clip, and clips are listed in train/validation files (80/20 split).
- **Model:** ImageNet-pretrained MobileNetV2 with a unidirectional temporal shift (`n_div=8`) inside its blocks. The shift only moves features from past to future frames, so the model can run on a live stream. Each clip is sampled into 12 segments.
- **Training:** SGD with momentum, class-weighted cross-entropy for the class imbalance, `ReduceLROnPlateau`, and early stopping (patience 5). The checkpoint with the best validation accuracy is saved.
- **Variants:** `action.ipynb` uses whole-frame clips; `action_per_person.ipynb` uses per-person clips, a larger dataset, and gradient accumulation (batch 16 x 4 steps).
- Each notebook ends with a confusion matrix on the validation set.

### Object detection (YOLO)

`object.ipynb` downloads the dataset (YOLO format with `data.yaml`) and fine-tunes `yolo26n.pt` with Ultralytics: 30 epochs, patience 3, image size 640, batch 48, on an NVIDIA GeForce RTX 3080 (10 GB).

## Tech Stack

- **Deep learning:** PyTorch, torchvision, Ultralytics YOLO
- **Data and vision:** OpenCV, NumPy, pandas, scikit-learn, Albumentations
- **Visualization:** Matplotlib, seaborn
- **Environment:** Docker with NVIDIA GPU (`pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`), JupyterLab

## Project Structure

```
concert-surveilance/
├── action.ipynb                 # TSM training on whole-frame clips
├── action_per_person.ipynb      # TSM training on per-person clips
├── object.ipynb                 # YOLO26n training (person, knife, weapon, hug)
├── ops/                         # TSM code (dataset, models, temporal_shift, transforms)
├── ops_per_person/              # Same modules for the per-person variant
├── models/                      # tsm_best.pth, training_log.csv
├── models_per_person/           # tsm_best_per_person.pth, training_log_per_person.csv
├── docker-compose.yml           # GPU JupyterLab environment
└── requirements.txt
```

The video datasets (`dataset/`), YOLO runs (`runs/`), and raw videos (`videos/`) are not stored in the repository.

## Getting Started

Requirements: an NVIDIA GPU with drivers and the NVIDIA Container Toolkit, plus Docker Compose.

```bash
git clone https://github.com/harrymardika/concert-surveilance.git
cd concert-surveilance
```

Replace `YOUR_SECURE_TOKEN` in `docker-compose.yml` with your own Jupyter token, then run:

```bash
docker compose up
```

JupyterLab runs at `http://localhost:8888` and installs `requirements.txt` on startup. Without Docker, use a virtual environment, install PyTorch, then `pip install -r requirements.txt`.

**Data:** for action recognition, put videos in `normal/` and `violence/` folders under the path set in the preprocessing cell, then run the frame extraction and list generation cells. `object.ipynb` downloads its dataset from Google Drive with `gdown`.

## Related Project

- [tsm-grid-camera](https://github.com/harrymardika/tsm-grid-camera): the real-time pipeline that combines grid-based TSM, YOLO, and ByteTrack.

## Author

**Harry Mardika** · [GitHub](https://github.com/harrymardika)
