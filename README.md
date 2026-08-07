# speed-watch

Car detection & tracking for racing footage — YOLOv8 + ByteTrack. v1 scope is detection and tracking only; downstream analytics (lap times, speed, racing line) are future work.

## Setup

```
pip install -r requirements.txt
```

Python 3.10+. Apple Silicon (MPS) is used automatically for training if available; inference runs fine on CPU.

## Structure

```
config.py              # paths and thresholds, single source of truth
main.py                 # CLI: run detection + tracking on a video clip end-to-end
pipeline/                # reusable detection/tracking code
  detector.py             CarDetector — pretrained YOLOv8 or fine-tuned checkpoint
  tracker.py               CarTracker — ByteTrack wrapper
  video_io.py               streaming frame I/O, H.264 video writer
  annotation.py              box/label drawing, CSV tracking log
training_notebooks/
  racing_vision_detection.ipynb   # experiments: Track A/B, comparison, tracking demo
data/roboflow_dataset/    # Roboflow YOLOv8-format training set (aerial/top-down, single class `car`)
videos/                    # input clips (gitignored)
models/                     # weights, incl. fine-tuned checkpoint (gitignored)
output_videos/               # annotated frames/video + tracking log CSVs
```

## Usage

Run the full pipeline on a clip:

```
python main.py videos/racing_clip1.mp4
```

Writes an annotated video and a per-frame tracking log CSV to `output_videos/`. Pass `--weights <path>` to use a fine-tuned checkpoint instead of pretrained YOLOv8n.

For the experiment writeup (detector comparison, training curves, tracking analysis), see `training_notebooks/racing_vision_detection.ipynb`. For v1 results and known limitations, see `NOTES.md`.
