# Sharp Object Detection — FIAP VisionGuard MVP

> **FIAP Pós-Tech · Tech Challenge 05**  
> Computer Vision · YOLOv8 · Supervised Object Detection

---

## Overview

This project is an MVP developed for **FIAP VisionGuard**, a security camera monitoring company.  
The goal is to validate the feasibility of using AI to automatically detect **sharp objects** (knives, scissors, and similar items) in security footage and trigger alerts to a security operations center.

The solution uses a **YOLOv8** model fine-tuned on a labeled dataset containing sharp objects under varying angles and lighting conditions, including negative samples to reduce false positives.

---

## Features

- Fine-tunes YOLOv8 on a custom sharp-object dataset
- Runs real-time detection on video files with bounding boxes and confidence scores
- Prints inline alerts to stdout for every detection
- Supports headless (no-display) mode for server environments
- CLI interface with configurable confidence threshold

---

## Project Structure

```
.
├── train.py        # Fine-tune YOLOv8 on the sharp-object dataset
├── detect.py       # Run detection on a video file
├── data.yaml       # Dataset configuration (paths + class names)
├── best.pt         # Trained model weights (after training)
└── requirements.txt
```

---

## Setup

### 1. Create and activate the Conda environment

```bash
conda create -n sharp-object-detection python=3.10 -y
conda activate sharp-object-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset

This project uses the **[CCTV Knife Detection Dataset](https://universe.roboflow.com/simuletic/cctv-knife-detection-dataset-zkkaf)** by Simuletic, available on Roboflow Universe.

| Property | Value |
|---|---|
| Source | Roboflow Universe / Kaggle |
| Images | 1 000 + synthetic CCTV-style frames |
| Classes | `knife`, `person` |
| Scenarios | Airports, bus stations, corridors, parking lots |
| Format | YOLOv8 (normalized bounding boxes) |

The synthetic images simulate real CCTV camera angles (elevated, wide-angle, corridor views) under varying lighting conditions, making it well suited for security-camera threat detection.

### Download the dataset

**Option A — Roboflow API (recommended)**

Get a free API key at [roboflow.com](https://roboflow.com), then run:

```bash
python download_dataset.py --api-key YOUR_ROBOFLOW_API_KEY
```

The script downloads the dataset in YOLOv8 format to `./dataset/` and prints the path to `data.yaml`.

**Option B — Kaggle**

```bash
kaggle datasets download -d simuletic/cctv-knife-detection-dataset
unzip cctv-knife-detection-dataset.zip -d dataset/
```

After downloading, update `data.yaml` to point to your local paths:

```yaml
train: ./dataset/train/images
val:   ./dataset/valid/images
test:  ./dataset/test/images

nc: 2
names: ['knife', 'person']
```

---

## Training

```bash
python train.py --epochs 50 --imgsz 640 --model yolov8n.pt
```

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 50 | Number of training epochs |
| `--imgsz` | 640 | Input image resolution |
| `--model` | `yolov8n.pt` | YOLOv8 base model (n/s/m/l/x) |
| `--data` | `data.yaml` | Dataset config path |
| `--name` | `sharp_object_model` | Run name under `runs/detect/` |

Trained weights will be saved at `runs/detect/<name>/weights/best.pt`.  
Copy or symlink `best.pt` to the project root before running detection.

---

## Detection

```bash
python detect.py --video video.mp4 --weights best.pt --conf 0.5
```

| Argument | Default | Description |
|---|---|---|
| `--video` | *(required)* | Path to input video file |
| `--weights` | `best.pt` | Path to trained weights |
| `--conf` | `0.5` | Confidence threshold (0–1) |
| `--no-display` | off | Disable preview window (headless) |

### Example output

```
[INFO] Loading model: best.pt
[INFO] Processing video: video.mp4
  [ALERT] Sharp object detected — frame 42, conf=0.87
  [ALERT] Sharp object detected — frame 43, conf=0.91
==================================================
[ALERT] 2 sharp object(s) detected in the video.
==================================================
```

Press **q** to stop the preview window early.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) | Object detection model |
| [OpenCV](https://opencv.org/) | Video capture and frame rendering |
| [PyTorch](https://pytorch.org/) | Deep learning backend |

---

## License

MIT
