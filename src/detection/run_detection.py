"""Run player/ball detection on a video and cache results to Parquet.

Usage:
    python -m src.detection.run_detection data/raw/clip.mp4 \
        --out data/processed/clip_detections.parquet --stride 3
"""

import argparse
from pathlib import Path

import cv2
import pandas as pd
import torch
from ultralytics import YOLO

# COCO class ids we care about: 0 = person, 32 = sports ball
TARGET_CLASSES = {0: "person", 32: "ball"}


def pick_device() -> str:
    """Return the best available device: Apple GPU (mps) if present, else CPU."""
    return "mps" if torch.backends.mps.is_available() else "cpu"


def run_detection(video_path: Path, out_path: Path, stride: int = 3,
                  weights: str = "yolov8s.pt", conf: float = 0.25) -> pd.DataFrame:
    """Detect people and balls in a video and save the results.

    Args:
        video_path: Path to the input video file.
        out_path: Where to write the Parquet file of detections.
        stride: Process every Nth frame (3 on 30 fps video ~ 10 fps).
            Higher values run faster, which matters on a laptop.
        weights: YOLO weights file; downloaded automatically on first use.
        conf: Minimum confidence score to keep a detection.

    Returns:
        A DataFrame with one row per detection and columns:
        frame, label, confidence, x1, y1, x2, y2.
    """
    model = YOLO(weights)
    device = pick_device()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    rows, frame_idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break  # end of video
        if frame_idx % stride == 0:
            result = model(frame, conf=conf, device=device,
                           classes=list(TARGET_CLASSES), verbose=False)[0]
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                rows.append({
                    "frame": frame_idx,
                    "label": TARGET_CLASSES[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                })
        frame_idx += 1
    cap.release()

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=3)
    args = parser.parse_args()

    df = run_detection(args.video, args.out, args.stride)
    print(f"Saved {len(df)} detections to {args.out}")