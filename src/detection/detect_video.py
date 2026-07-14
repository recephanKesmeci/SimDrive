"""Detect vehicles, pedestrians and motorcycles in a video with YOLOv8.

Reads an input video, runs YOLOv8 on every frame, draws boxes for the
traffic-relevant COCO classes and writes the annotated result next to the
input file (or to --output).

Usage:
    python src/detection/detect_video.py data/traffic.mp4
    python src/detection/detect_video.py data/traffic.mp4 -o out.mp4 --conf 0.4 --show
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

# COCO class ids we care about for traffic scenes
TARGET_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# BGR colors per class id
CLASS_COLORS = {
    0: (0, 220, 0),      # person: green
    1: (255, 180, 0),    # bicycle: light blue
    2: (0, 140, 255),    # car: orange
    3: (200, 0, 255),    # motorcycle: pink
    5: (255, 0, 100),    # bus: purple-blue
    7: (0, 215, 255),    # truck: yellow
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLOv8 vehicle/pedestrian detection on a video file."
    )
    parser.add_argument("video", type=Path, help="Path to the input video file")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output video path (default: <input>_annotated.mp4)",
    )
    parser.add_argument(
        "--model", default="yolov8n.pt",
        help="Ultralytics model name or path (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--conf", type=float, default=0.3,
        help="Confidence threshold (default: 0.3)",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Also display the annotated frames in a window while processing",
    )
    return parser.parse_args()


def draw_detections(frame, boxes) -> int:
    """Draw target-class boxes on the frame in place. Returns count drawn."""
    drawn = 0
    for box in boxes:
        cls_id = int(box.cls[0])
        if cls_id not in TARGET_CLASSES:
            continue
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
        conf = float(box.conf[0])
        color = CLASS_COLORS[cls_id]
        label = f"{TARGET_CLASSES[cls_id]} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        drawn += 1
    return drawn


def main() -> None:
    args = parse_args()

    if not args.video.is_file():
        sys.exit(f"Input video not found: {args.video}")

    output_path = args.output or args.video.with_name(
        f"{args.video.stem}_annotated.mp4"
    )

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        sys.exit(f"Could not open output for writing: {output_path}")

    print(f"Processing {args.video.name}: {width}x{height} @ {fps:.1f} fps, "
          f"{total_frames} frames")

    frame_idx = 0
    total_detections = 0
    start = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model(frame, conf=args.conf, verbose=False)
            total_detections += draw_detections(frame, results[0].boxes)
            writer.write(frame)

            if args.show:
                cv2.imshow("SimDrive detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Interrupted by user (q pressed)")
                    break

            frame_idx += 1
            if frame_idx % 50 == 0:
                elapsed = time.perf_counter() - start
                print(f"  frame {frame_idx}/{total_frames} "
                      f"({frame_idx / elapsed:.1f} fps)")
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start
    print(f"Done: {frame_idx} frames in {elapsed:.1f}s "
          f"({frame_idx / max(elapsed, 1e-6):.1f} fps), "
          f"{total_detections} detections drawn")
    print(f"Annotated video saved to: {output_path}")


if __name__ == "__main__":
    main()
