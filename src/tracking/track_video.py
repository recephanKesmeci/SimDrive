"""Track vehicles, pedestrians and motorcycles in a video with YOLOv8 + ByteTrack.

Runs Ultralytics' built-in ByteTrack tracker on every frame, draws boxes with
persistent track IDs and a motion trail per object, and writes the annotated
result next to the input file (or to --output).

Usage:
    python src/tracking/track_video.py data/traffic.mp4
    python src/tracking/track_video.py data/traffic.mp4 -o out.mp4 --conf 0.4 --show
"""

import argparse
import sys
import time
from collections import defaultdict, deque
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

# BGR colors cycled by track id so neighboring IDs look distinct
ID_COLORS = [
    (0, 220, 0), (0, 140, 255), (255, 180, 0), (200, 0, 255),
    (255, 0, 100), (0, 215, 255), (255, 100, 100), (100, 255, 255),
]

TRAIL_LENGTH = 30  # number of past box centers kept per track


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLOv8 + ByteTrack multi-object tracking on a video file."
    )
    parser.add_argument("video", type=Path, help="Path to the input video file")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output video path (default: <input>_tracked.mp4)",
    )
    parser.add_argument(
        "--model", default="yolov8n.pt",
        help="Ultralytics model name or path (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--conf", type=float, default=0.3,
        help="Detection confidence threshold (default: 0.3)",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Also display the annotated frames in a window while processing",
    )
    return parser.parse_args()


def draw_tracks(frame, boxes, trails) -> set:
    """Draw tracked boxes, IDs and trails in place. Returns active track ids."""
    active_ids = set()
    if boxes.id is None:  # tracker had nothing to associate this frame
        return active_ids

    for box in boxes:
        cls_id = int(box.cls[0])
        if cls_id not in TARGET_CLASSES:
            continue
        track_id = int(box.id[0])
        active_ids.add(track_id)
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
        color = ID_COLORS[track_id % len(ID_COLORS)]
        label = f"#{track_id} {TARGET_CLASSES[cls_id]}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Trail: line through recent box-bottom centers (ground contact point)
        trail = trails[track_id]
        trail.append(((x1 + x2) // 2, y2))
        for a, b in zip(trail, list(trail)[1:]):
            cv2.line(frame, a, b, color, 2)

    return active_ids


def main() -> None:
    args = parse_args()

    if not args.video.is_file():
        sys.exit(f"Input video not found: {args.video}")

    output_path = args.output or args.video.with_name(
        f"{args.video.stem}_tracked.mp4"
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

    trails = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))
    all_ids = set()
    frame_idx = 0
    start = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # persist=True keeps tracker state between frames (required for video)
            results = model.track(
                frame, conf=args.conf, persist=True,
                tracker="bytetrack.yaml", verbose=False,
            )
            active_ids = draw_tracks(frame, results[0].boxes, trails)
            all_ids |= active_ids
            writer.write(frame)

            if args.show:
                cv2.imshow("SimDrive tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Interrupted by user (q pressed)")
                    break

            frame_idx += 1
            if frame_idx % 50 == 0:
                elapsed = time.perf_counter() - start
                print(f"  frame {frame_idx}/{total_frames} "
                      f"({frame_idx / elapsed:.1f} fps, "
                      f"{len(all_ids)} unique tracks)")
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start
    print(f"Done: {frame_idx} frames in {elapsed:.1f}s "
          f"({frame_idx / max(elapsed, 1e-6):.1f} fps), "
          f"{len(all_ids)} unique objects tracked")
    print(f"Tracked video saved to: {output_path}")


if __name__ == "__main__":
    main()
