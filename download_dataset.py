"""
download_dataset.py
-------------------
Downloads the CCTV Knife Detection Dataset from Roboflow Universe.

Dataset: https://universe.roboflow.com/simuletic/cctv-knife-detection-dataset-zkkaf
- Synthetic CCTV-style images (indoor/outdoor, bus stations, airports, corridors)
- Classes: knife, person
- Annotation format: YOLOv8 (normalized bounding boxes)

Requirements:
    pip install roboflow

Usage:
    python download_dataset.py --api-key YOUR_ROBOFLOW_API_KEY

How to get a free API key:
    1. Create a free account at https://roboflow.com
    2. Go to your workspace settings → API keys
    3. Copy your Private API Key

The dataset will be downloaded to ./dataset/ and data.yaml will be updated
to point to the correct paths automatically.
"""

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the CCTV Knife Detection Dataset")
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="Your Roboflow API key (get it at https://roboflow.com)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset",
        help="Directory where the dataset will be saved (default: ./dataset)",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="Dataset version to download (default: 1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from roboflow import Roboflow
    except ImportError:
        print("[ERROR] roboflow package not found. Run: pip install roboflow", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("[INFO] Connecting to Roboflow...")
    rf = Roboflow(api_key=args.api_key)

    print("[INFO] Fetching dataset: simuletic / cctv-knife-detection-dataset-zkkaf")
    project = rf.workspace("simuletic").project("cctv-knife-detection-dataset-zkkaf")
    dataset = project.version(args.version).download(
        model_format="yolov8",
        location=str(output_path),
        overwrite=True,
    )

    print(f"[INFO] Dataset downloaded to: {output_path.resolve()}")
    print(f"[INFO] data.yaml location:    {output_path / 'data.yaml'}")
    print("\n[NEXT STEP] Run training:")
    print(f"  python train.py --data {output_path / 'data.yaml'}")


if __name__ == "__main__":
    main()
