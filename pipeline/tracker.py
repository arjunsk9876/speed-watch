import supervision as sv


class CarTracker:
    """Thin wrapper around supervision's ByteTrack for persistent car IDs."""

    def __init__(self):
        self.tracker = sv.ByteTrack()

    def update(self, detections: sv.Detections) -> sv.Detections:
        return self.tracker.update_with_detections(detections)

    def reset(self):
        self.tracker.reset()
