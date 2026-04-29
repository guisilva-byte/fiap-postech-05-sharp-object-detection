"""
train.py
--------
Fine-tunes a YOLOv8 model on the sharp-object dataset defined in data.yaml.

Usage:
    python train.py [--epochs N] [--imgsz S] [--model MODEL]

The trained weights are saved under runs/detect/<name>/weights/best.pt.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 for sharp-object detection")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 base model")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to dataset YAML")
    parser.add_argument("--name", type=str, default="sharp_object_model", help="Run name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    print(f"[INFO] Loading base model: {args.model}")
    model = YOLO(args.model)

    print(f"[INFO] Starting training — epochs={args.epochs}, imgsz={args.imgsz}")
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        name=args.name,
    )

    best_weights = Path("runs") / "detect" / args.name / "weights" / "best.pt"
    print(f"[INFO] Training complete. Best weights saved at: {best_weights}")
    return results


if __name__ == "__main__":
    main()
