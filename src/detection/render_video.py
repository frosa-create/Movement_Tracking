"""Draw cached detections onto a video so you can inspect them visually.

Usage:
    python -m src.detection.render_video data/raw/clip.mp4 \
        data/processed/clip_detections.parquet \
        --out data/processed/clip_annotated.mp4 --stride 3
"""

import argparse
from pathlib import Path

import cv2
import pandas as pd

# BGR colors per label so players and balls are easy to tell apart
COLORS = {"person": (0, 255, 0), "ball": (0, 0, 255)}


def render(video_path: Path, detections_path: Path, out_path: Path,
           stride: int = 3) -> None:
    """Write a copy of the video with detection boxes drawn on it.

    Only frames that were processed during detection (every `stride`-th
    frame) are written, so the output plays at fps / stride to keep
    the original speed.

    Args:
        video_path: Original input video.
        detections_path: Parquet file produced by run_detection.
        out_path: Where to save the annotated video (.mp4).
        stride: Must match the stride used when detecting.
    """
    df = pd.read_parquet(detections_path)
    by_frame = {f: g for f, g in df.groupby("frame")}  # fast per-frame lookup

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / stride, (width, height))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            for row in by_frame.get(frame_idx, pd.DataFrame()).itertuples():
                color = COLORS.get(row.label, (255, 255, 255))
                p1, p2 = (int(row.x1), int(row.y1)), (int(row.x2), int(row.y2))
                cv2.rectangle(frame, p1, p2, color, 2)
            writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("detections", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=3)
    args = parser.parse_args()
    render(args.video, args.detections, args.out, args.stride)
    print(f"Saved annotated video to {args.out}")