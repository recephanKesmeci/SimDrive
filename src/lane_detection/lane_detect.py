"""Classical lane detection on a video: Canny edges + Hough line transform.

No deep learning — pure image processing. Pipeline per frame:
grayscale -> Gaussian blur -> Canny -> trapezoidal ROI mask -> HoughLinesP
-> filter segments by slope -> average into one left and one right lane line
-> temporal smoothing -> overlay.

Designed for forward-facing (dashcam-style) footage where lanes converge
toward the horizon. Tune ROI_* constants for other camera angles.

Usage:
    python src/lane_detection/lane_detect.py data/dashcam.mp4
    python src/lane_detection/lane_detect.py data/dashcam.mp4 --debug --show
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ROI trapezoid as fractions of frame size (forward-facing camera assumption)
ROI_TOP_Y = 0.60        # top edge of ROI, fraction of height
ROI_TOP_X_MARGIN = 0.42  # left/right inset of ROI top edge, fraction of width
ROI_BOTTOM_X_MARGIN = 0.05

MIN_SLOPE = 0.4         # reject near-horizontal Hough segments
SMOOTH_ALPHA = 0.2      # EMA factor for temporal smoothing of lane lines

LANE_COLOR = (0, 255, 0)      # BGR
FILL_COLOR = (0, 120, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classical (Canny + Hough) lane detection on a video file."
    )
    parser.add_argument("video", type=Path, help="Path to the input video file")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output video path (default: <input>_lanes.mp4)",
    )
    parser.add_argument(
        "--canny", type=int, nargs=2, default=(50, 150), metavar=("LOW", "HIGH"),
        help="Canny thresholds (default: 50 150)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Render edge map and raw Hough segments in a side strip",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Also display the annotated frames in a window while processing",
    )
    return parser.parse_args()


def roi_vertices(width: int, height: int) -> np.ndarray:
    top_y = int(height * ROI_TOP_Y)
    return np.array([[
        (int(width * ROI_BOTTOM_X_MARGIN), height),
        (int(width * ROI_TOP_X_MARGIN), top_y),
        (int(width * (1 - ROI_TOP_X_MARGIN)), top_y),
        (int(width * (1 - ROI_BOTTOM_X_MARGIN)), height),
    ]], dtype=np.int32)


def detect_segments(frame, canny_low: int, canny_high: int):
    """Return (Hough segments in ROI, edge map) for a BGR frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)

    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices(frame.shape[1], frame.shape[0]), 255)
    masked = cv2.bitwise_and(edges, mask)

    segments = cv2.HoughLinesP(
        masked, rho=2, theta=np.pi / 180, threshold=50,
        minLineLength=40, maxLineGap=100,
    )
    # normalize to (N, 4): OpenCV 4 returns (N, 1, 4), OpenCV 5 returns (N, 4)
    if segments is not None:
        segments = segments.reshape(-1, 4)
    return segments, masked


def average_lane(segments, side: str, height: int):
    """Fit one lane line from Hough segments of one side.

    Averages segment (slope, intercept) weighted by segment length; returns
    endpoints [(x_bottom, y_bottom), (x_top, y_top)] or None.
    """
    fits, weights = [], []
    for seg in segments:
        x1, y1, x2, y2 = seg
        if x1 == x2:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < MIN_SLOPE:
            continue
        if (side == "left") != (slope < 0):  # left lane has negative slope
            continue
        fits.append((slope, y1 - slope * x1))
        weights.append(np.hypot(x2 - x1, y2 - y1))

    if not fits:
        return None
    slope, intercept = np.average(fits, axis=0, weights=weights)
    y_bottom, y_top = height, int(height * ROI_TOP_Y)
    x_bottom = int((y_bottom - intercept) / slope)
    x_top = int((y_top - intercept) / slope)
    return np.array([x_bottom, y_bottom, x_top, y_top], dtype=float)


def draw_lanes(frame, left, right) -> None:
    """Overlay lane lines and the lane area fill in place."""
    overlay = frame.copy()
    for lane in (left, right):
        if lane is None:
            continue
        x1, y1, x2, y2 = (int(v) for v in lane)
        cv2.line(overlay, (x1, y1), (x2, y2), LANE_COLOR, 8)
    if left is not None and right is not None:
        quad = np.array([[
            (int(left[0]), int(left[1])), (int(left[2]), int(left[3])),
            (int(right[2]), int(right[3])), (int(right[0]), int(right[1])),
        ]], dtype=np.int32)
        cv2.fillPoly(overlay, quad, FILL_COLOR)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)


def main() -> None:
    args = parse_args()

    if not args.video.is_file():
        sys.exit(f"Input video not found: {args.video}")

    output_path = args.output or args.video.with_name(
        f"{args.video.stem}_lanes.mp4"
    )

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

    smooth = {"left": None, "right": None}
    frame_idx = 0
    detected_frames = 0
    start = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            segments, edges = detect_segments(frame, *args.canny)
            lanes = {}
            for side in ("left", "right"):
                lane = (average_lane(segments, side, height)
                        if segments is not None else None)
                # exponential moving average keeps lines stable across frames
                if lane is not None:
                    if smooth[side] is None:
                        smooth[side] = lane
                    else:
                        smooth[side] = (SMOOTH_ALPHA * lane
                                        + (1 - SMOOTH_ALPHA) * smooth[side])
                lanes[side] = smooth[side]

            if lanes["left"] is not None or lanes["right"] is not None:
                detected_frames += 1
            draw_lanes(frame, lanes["left"], lanes["right"])
            cv2.polylines(frame, roi_vertices(width, height), True,
                          (180, 180, 180), 1)

            if args.debug:
                strip = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                if segments is not None:
                    for seg in segments:
                        x1, y1, x2, y2 = seg
                        cv2.line(strip, (x1, y1), (x2, y2), (0, 0, 255), 2)
                strip = cv2.resize(strip, (width // 4, height // 4))
                frame[0:height // 4, width - width // 4:width] = strip

            writer.write(frame)

            if args.show:
                cv2.imshow("SimDrive lanes", frame)
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
          f"({frame_idx / max(elapsed, 1e-6):.1f} fps), lanes found in "
          f"{detected_frames}/{frame_idx} frames")
    print(f"Lane video saved to: {output_path}")


if __name__ == "__main__":
    main()
