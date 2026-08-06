import csv

import supervision as sv

_box_annotator = sv.BoxAnnotator()
_label_annotator = sv.LabelAnnotator()


def annotate_frame(frame, detections: sv.Detections):
    """Draw boxes + track-id labels onto a copy of the frame."""
    labels = [
        f"#{tracker_id} {confidence:.2f}" if tracker_id is not None else f"{confidence:.2f}"
        for tracker_id, confidence in zip(
            detections.tracker_id if detections.tracker_id is not None else [None] * len(detections),
            detections.confidence if detections.confidence is not None else [0.0] * len(detections),
        )
    ]
    annotated = _box_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated = _label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
    return annotated


class TrackingLogger:
    """Accumulates per-frame detection/tracking rows and writes them to CSV."""

    def __init__(self):
        self.rows = []

    def log(self, frame_idx, detections: sv.Detections):
        tracker_ids = detections.tracker_id if detections.tracker_id is not None else [None] * len(detections)
        confidences = detections.confidence if detections.confidence is not None else [None] * len(detections)
        for (x1, y1, x2, y2), tracker_id, confidence in zip(detections.xyxy, tracker_ids, confidences):
            self.rows.append(
                {
                    "frame": frame_idx,
                    "track_id": int(tracker_id) if tracker_id is not None else "",
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "confidence": float(confidence) if confidence is not None else "",
                }
            )

    def write_csv(self, output_path):
        fieldnames = ["frame", "track_id", "x1", "y1", "x2", "y2", "confidence"]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
