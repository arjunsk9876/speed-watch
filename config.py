from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

VIDEOS_DIR = ROOT_DIR / "videos"
ROBOFLOW_DATA_YAML = ROOT_DIR / "data" / "roboflow_dataset" / "data.yaml"
MODELS_DIR = ROOT_DIR / "models"
OUTPUT_VIDEOS_DIR = ROOT_DIR / "output_videos"

PRETRAINED_WEIGHTS = MODELS_DIR / "yolov8n.pt"
FINETUNED_WEIGHTS = MODELS_DIR / "finetune" / "weights" / "best.pt"

DETECTION_CONF = 0.25
