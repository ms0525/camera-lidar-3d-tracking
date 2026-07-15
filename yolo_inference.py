# SPDX-License-Identifier: AGPL-3.0-only
"""Run a configurable Ultralytics YOLO 2D-detection demo on one image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

from utils.calib_loader import load_calibration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Ultralytics YOLO on a KITTI camera image.")
    parser.add_argument(
        "--image",
        required=True,
        help="Input camera image supplied by the user",
    )
    parser.add_argument("--model", default="yolo26s.pt")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--calib", help="Optional KITTI calibration file whose P2 is printed")
    parser.add_argument("--output", help="Optional path for the annotated BGR image")
    parser.add_argument("--no-show", action="store_true", help="Do not open a Matplotlib window")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    if args.calib:
        calibration = load_calibration(args.calib)
        print("Projection matrix P2:\n", calibration["P2"].reshape(3, 4))

    # Ultralytics expects OpenCV/NumPy inputs in BGR order.
    model = YOLO(args.model)
    result = model.predict(image_bgr, conf=args.confidence, verbose=False)[0]
    annotated_bgr = result.plot()

    print(f"Detected {len(result.boxes)} objects in {image_path}")
    for box in result.boxes:
        class_name = model.names[int(box.cls[0].item())]
        confidence = float(box.conf[0].item())
        print(f"  {class_name}: {confidence:.3f}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), annotated_bgr):
            raise OSError(f"Could not write annotated image: {output_path}")
        print(f"Saved {output_path}")

    if not args.no_show:
        plt.imshow(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("YOLO Detection on KITTI")
        plt.tight_layout()
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
