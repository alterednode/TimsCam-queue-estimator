from multiprocessing import freeze_support
from pathlib import Path
from ultralytics import YOLO


def main() -> None:
    #change the model here to export different models
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model = YOLO("yolov8s.pt")
    model.export(format="openvino", dynamic=True, half=True, project=str(models_dir))


if __name__ == "__main__":
    freeze_support()
    main()
