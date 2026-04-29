"""
detect.py
---------
Runs sharp-object detection on a video file using a trained YOLOv8 model.

Detections are drawn on each frame in real time. A summary alert is printed
at the end showing the total number of detections found.

Usage:
    python detect.py --video <path> [--weights <path>] [--conf 0.5]

Press 'q' to quit the preview window early.
"""

import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO


# BGR color used for bounding boxes and labels
_BOX_COLOR = (0, 255, 0)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sharp-object detection on video")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument(
        "--weights", type=str, default="best.pt", help="Path to trained YOLO weights"
    )
    parser.add_argument(
        "--conf", type=float, default=0.5, help="Confidence threshold (0–1)"
    )
    parser.add_argument(
        "--no-display", action="store_true", help="Disable the preview window (headless mode)"
    )
    return parser.parse_args()


def draw_detection(frame, box, conf: float) -> None:
    """Draw a bounding box and confidence label on *frame* in place."""
    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
    cv2.rectangle(frame, (x1, y1), (x2, y2), _BOX_COLOR, 2)

    label = f"sharp {conf:.2f}"
    cv2.putText(frame, label, (x1, y1 - 10), _FONT, 0.6, _BOX_COLOR, 2)


def run(video_path: str, weights: str, conf_thresh: float, display: bool) -> int:
    """
    Process the video and return the total number of sharp-object detections.

    Parameters
    ----------
    video_path : str
        Path to the input video.
    weights : str
        Path to YOLOv8 weights file (.pt).
    conf_thresh : float
        Minimum confidence to count a detection.
    display : bool
        Whether to open a preview window.

    Returns
    -------
    int
        Total detection count across all frames.
    """
    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"Video file not found: {video}")

    print(f"[INFO] Loading model: {weights_path}")
    model = YOLO(str(weights_path))

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    total_detections = 0
    frame_count = 0

    print(f"[INFO] Processing video: {video}")
    print("[INFO] Press 'q' to stop the preview window.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = model.predict(frame, conf=conf_thresh, verbose=False)

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                draw_detection(frame, box, conf)
                total_detections += 1

                # Inline alert: log each detection to stdout
                print(f"  [ALERT] Sharp object detected — frame {frame_count}, conf={conf:.2f}")

        if display:
            cv2.imshow("Sharp Object Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] User interrupted — stopping early.")
                break

    cap.release()
    if display:
        cv2.destroyAllWindows()

    return total_detections


def main() -> None:
    args = parse_args()

    try:
        total = run(
            video_path=args.video,
            weights=args.weights,
            conf_thresh=args.conf,
            display=not args.no_display,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 50)
    if total > 0:
        print(f"[ALERT] {total} sharp object(s) detected in the video.")
    else:
        print("[OK] No sharp objects detected.")
    print("=" * 50)


if __name__ == "__main__":
    main()
