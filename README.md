# SimDrive

Sim-to-real perception pipeline for traffic scenes: real-video object detection,
tracking and lane detection, combined with synthetic data generation in Unity to
study how well models fine-tuned on simulated data transfer to real footage.

Built as preparation for UGV navigation/perception work (robotics internship, 2027).

## Project Vision (6-week roadmap)

| Week | Milestone |
|------|-----------|
| 1 | Vehicle/pedestrian detection on dashcam & traffic video with YOLOv8 |
| 2 | Multi-object tracking with ByteTrack (persistent IDs) + classical lane detection (Canny + Hough) |
| 3 | Distance estimation, approach/collision warning, Streamlit dashboard |
| 4 | Synthetic traffic data generation with Unity Perception (domain randomization) |
| 5 | YOLOv8 fine-tuning on the synthetic dataset |
| 6 | Experiment: pretrained COCO weights vs. synthetic fine-tuned weights on real video |

## Repository Layout

```
src/
  detection/        # YOLOv8 detection scripts
  tracking/         # ByteTrack multi-object tracking
  lane_detection/   # Classical lane detection (Canny + Hough)
unity_project/      # Unity Perception synthetic data project
experiments/        # Evaluation runs, comparisons, notebooks
data/               # Videos & datasets (git-ignored)
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

Detect vehicles, pedestrians and motorcycles in a video and save an annotated copy:

```powershell
python src/detection/detect_video.py data/traffic.mp4
# optional flags:
python src/detection/detect_video.py data/traffic.mp4 -o data/out.mp4 --model yolov8n.pt --conf 0.3 --show
```

## Hardware Notes

- HP Victus (Windows 11). Built-in webcam is broken; live-camera experiments use
  a phone camera exposed as a virtual webcam via iVCam.
