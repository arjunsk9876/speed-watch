# speed-watch

Car detection & tracking for racing footage — YOLOv8 + ByteTrack. v1 scope is detection and tracking only; downstream analytics (lap times, speed, racing line) are future work.

## Setup

```bash
git clone <this repo>
cd speed-watch
pip3 install -r requirements.txt
```

Requires Python 3.10+. On macOS, `python3`/`pip3` — not `python`/`pip`, which usually don't exist by default.

Training uses Apple Silicon MPS automatically if available (`torch.backends.mps.is_available()`); inference (running `main.py`, the notebook's Track A/B/comparison cells) runs fine on CPU, just slower.

Input videos are not checked into git (`videos/*.mp4` is gitignored) — the two racing clips need to already be in `videos/` for anything below to run. Same for `models/*.pt` (weights) and `models/finetune/` (training run artifacts) — those regenerate from `training_notebooks/racing_vision_detection.ipynb`'s Track A/B cells if missing.

## Structure

```
config.py                    # paths and thresholds, single source of truth
main.py                      # CLI: run detection + tracking on a video clip end-to-end
pipeline/                    # reusable detection/tracking code
  detector.py                  CarDetector — pretrained YOLOv8 or fine-tuned checkpoint
  tracker.py                   CarTracker — ByteTrack wrapper
  video_io.py                  streaming frame I/O, H.264 video writer
  annotation.py                box/label drawing, CSV tracking log
training_notebooks/
  racing_vision_detection.ipynb   # experiments: Track A/B, comparison, tracking demo
data/roboflow_dataset/       # Roboflow YOLOv8-format training set (aerial/top-down, single class `car`)
videos/                      # input clips (gitignored — add your own .mp4 files here)
models/                      # weights, incl. fine-tuned checkpoint (gitignored)
output_videos/                # annotated frames/video + tracking log CSVs (committed as deliverables)
```

## Usage

### Run the full pipeline on a clip

```bash
python3 main.py videos/racing_clip1.mp4
```

Detects cars (pretrained YOLOv8n by default), tracks them with ByteTrack, and writes:
- `output_videos/racing_clip1_tracked.mp4` — annotated video (boxes + track IDs)
- `output_videos/racing_clip1_tracking_log.csv` — per-frame `frame,track_id,x1,y1,x2,y2,confidence`

Options:

```bash
python3 main.py videos/racing_clip2.mp4                                          # different clip
python3 main.py videos/racing_clip1.mp4 --weights models/finetune/weights/best.pt  # fine-tuned checkpoint instead of pretrained
python3 main.py videos/racing_clip1.mp4 --output-name custom_name.mp4             # override output filename
```

This processes every frame, so it's slow on a full clip — the reference run (34,133 frames, 640×360) took ~50 minutes on an M1 Pro CPU. Runtime scales with clip length; there's no built-in way to process only part of a clip via `main.py`.

### View existing results without running anything

```bash
open output_videos/racing_clip1_tracked.mp4   # already-generated annotated video
cat NOTES.md                                   # v1 results summary
head output_videos/racing_clip1_tracking_log.csv
```

### Open the experiment notebook

```bash
jupyter notebook training_notebooks/racing_vision_detection.ipynb
```

All cells already have saved output (detection comparisons, training curves, tracking stats) — no need to re-run anything just to read it. If you do re-run cells: Track A/B/comparison cells reuse `model_pretrained`/`model_finetuned` loaded earlier in the notebook, so run cells top-to-bottom, not out of order. The Track B training cell is idempotent — it checks `models/finetune/results.csv` for a completed 50-epoch run before retraining, so re-running the notebook won't accidentally kick off another ~50-minute fine-tune.

## Troubleshooting

**`objc[...]: Class AVFFrameReceiver is implemented in both ...`** — harmless. Two dependencies (`av` and `opencv-python`) each bundle their own copy of a video codec library; macOS warns about the duplicate symbol but nothing breaks. Ignorable.

**`FutureWarning: The 'ByteTrack' was deprecated...`** — harmless, from `supervision`. A future version renames the class; current version works fine.

**`command not found: python`** — use `python3` (and `pip3` for installs). macOS doesn't ship a bare `python` command.

**Training/inference cells produce no visible progress for a long time** — expected for full-clip runs; `main.py` and the tracking-demo notebook cell only print once processing finishes, not per-frame.

For v1 results, known limitations, and what changed from the original plan mid-build, see `NOTES.md`.
