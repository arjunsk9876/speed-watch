import argparse
from pathlib import Path

import config as cfg
from pipeline import CarDetector, CarTracker, TrackingLogger, annotate_frame, read_video, save_video


def run(video_path, weights=None, output_name=None):
    detector = (
        CarDetector.finetuned(weights)
        if weights
        else CarDetector.pretrained(weights=str(cfg.PRETRAINED_WEIGHTS))
    )
    tracker = CarTracker()
    logger = TrackingLogger()

    frames = read_video(video_path)
    annotated_frames = []
    for i, frame in enumerate(frames):
        detections = detector.predict(frame)
        detections = tracker.update(detections)
        logger.log(i, detections)
        annotated_frames.append(annotate_frame(frame, detections))

    cfg.OUTPUT_VIDEOS_DIR.mkdir(exist_ok=True)
    stem = Path(video_path).stem
    output_name = output_name or f"{stem}_tracked.mp4"

    video_out_path = cfg.OUTPUT_VIDEOS_DIR / output_name
    csv_out_path = cfg.OUTPUT_VIDEOS_DIR / f"{stem}_tracking_log.csv"

    save_video(annotated_frames, video_out_path)
    logger.write_csv(csv_out_path)

    print(f"Annotated video: {video_out_path}")
    print(f"Tracking log:    {csv_out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run car detection + ByteTrack tracking on a video clip.")
    parser.add_argument("video", nargs="?", default=str(cfg.VIDEOS_DIR / "racing_clip1.mp4"))
    parser.add_argument("--weights", default=None, help="Fine-tuned weights path; omit to use pretrained YOLOv8n")
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()

    run(args.video, weights=args.weights, output_name=args.output_name)
