import supervision as sv
from ultralytics import YOLO

# COCO class ids that correspond to ground vehicles
COCO_VEHICLE_CLASS_IDS = (2, 5, 7)  # car, bus, truck


class CarDetector:
    """Thin wrapper around a YOLOv8 model for car detection.

    Works with either the pretrained COCO checkpoint (filtered to vehicle
    classes) or a checkpoint fine-tuned on the single-class `car` dataset.
    """

    def __init__(self, weights, vehicle_class_ids=COCO_VEHICLE_CLASS_IDS, conf=0.25):
        self.model = YOLO(weights)
        self.vehicle_class_ids = list(vehicle_class_ids) if vehicle_class_ids else None
        self.conf = conf

    @classmethod
    def pretrained(cls, weights="models/yolov8n.pt", conf=0.25):
        return cls(weights=weights, vehicle_class_ids=COCO_VEHICLE_CLASS_IDS, conf=conf)

    @classmethod
    def finetuned(cls, weights, conf=0.25):
        return cls(weights=weights, vehicle_class_ids=None, conf=conf)

    def predict(self, frame) -> sv.Detections:
        result = self.model(frame, classes=self.vehicle_class_ids, conf=self.conf, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    @staticmethod
    def train(data_yaml, epochs=50, imgsz=640, project="models", name="finetune", base_weights="yolov8n.pt", device=None):
        """Fine-tune a pretrained YOLOv8 checkpoint on a Roboflow-format dataset."""
        model = YOLO(base_weights)
        return model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, project=project, name=name, device=device)
