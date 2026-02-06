from __future__ import annotations

import argparse
import sys
from multiprocessing import freeze_support
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a YOLO model to OpenVINO format")
    parser.add_argument("--model", default="yolov8s.pt", help="Path or name of YOLO model to export")
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directory where exported OpenVINO files will be written",
    )
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic input shape")
    parser.add_argument("--half", action="store_true", help="Export weights in FP16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "ultralytics is not installed. Install optional export tools with: uv sync --extra export",
            file=sys.stderr,
        )
        raise SystemExit(2)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    model.export(
        format="openvino",
        dynamic=args.dynamic,
        half=args.half,
        project=str(output_dir),
    )


if __name__ == "__main__":
    freeze_support()
    main()
