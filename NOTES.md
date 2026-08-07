# v1 Notes — Detection & Tracking

## Detector: pretrained wins

Two tracks were compared on the same sampled frames from both clips:

| | clip1 mean cars/frame | clip2 mean cars/frame |
|---|---|---|
| Pretrained YOLOv8n (COCO) | 2.83 | 0.17 |
| Fine-tuned on Roboflow set | 0.00 | 0.00 |

**Pretrained `yolov8n.pt` carries forward.** The Roboflow training set is aerial/top-down; both clips are ground-level/trackside. Fine-tuning on that mismatch didn't just fail to help — it made detection worse, dropping clip1 from a 2.83 mean to zero. The fine-tune converged well on its own validation split (mAP50 0.983, mAP50-95 0.851) but that skill didn't transfer; it looks like it overwrote some of the COCO-pretrained model's general "car" representation with an aerial-silhouette-specific one. This was a real risk flagged before training (see the notebook's angle-mismatch section) and it played out as expected.

Even the pretrained model is far from the PRD's ≥80%-detection target — clip2 barely registers cars at all (0.17 mean). Detection quality, not just the fine-tune, is the main open problem.

## Tracking: ByteTrack on full clip1 (34,133 frames, ~23 min)

- 48.8% of frames had at least one detected car.
- 681 distinct track IDs were assigned, against a per-frame max of ~6 cars actually visible. That gap is heavy ID fragmentation: ByteTrack has no re-identification model, so every time detection drops out for a frame or two the same physical car comes back as a new ID instead of the same one continuing.
- The PRD's ≥90% ID-persistence target is not met. Root cause looks like detection recall more than the tracker itself — a car that's never detected can't be tracked continuously no matter how good the tracker is. Fixing detection consistency would likely reduce fragmentation more than swapping trackers.
- Logged as a known v1 gap, not something fixed here.

## What changed vs. the original plan

- Repo restructured mid-build from a single notebook into a `pipeline/` package (detector, tracker, video I/O, annotation) + `main.py` entrypoint, with the notebook moved to `training_notebooks/` as the experiment/writeup surface. Reasoning: keep reusable logic out of notebook cells so `main.py` and future phases can call it directly.
- `VideoWriter` was switched from cv2's `mp4v` fourcc to PyAV/H.264 — the mp4v output was 120MB for the full clip1 tracking video, uncomfortably close to GitHub's 100MB file limit; H.264 brought the same content down to 57MB.
- `read_video`/`save_video` (load-entire-clip-into-memory helpers) were replaced with streaming `iter_frames`/`VideoWriter` — the original versions would have tried to hold ~24GB of frames in memory for the full 34k-frame clip1 run.
- Track B's training cell is idempotent: it checks `results.csv` epoch count before retraining, so re-running the notebook doesn't redo a 50-epoch fine-tune it doesn't need to. (An earlier version used `resume=True` on an interrupted checkpoint — that turned out to silently fall back to training a fresh model on the wrong default dataset instead of erroring, so resume was dropped in favor of always retraining from the pretrained base when incomplete.)
