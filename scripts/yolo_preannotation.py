"""Generate LabelMe rectangle annotations using a pretrained YOLO model.

The script detects people and sports balls, then writes a LabelMe-compatible
JSON file beside each image. It is intended for AI-assisted pre-annotation;
all generated labels should be reviewed and corrected by a human annotator.
"""

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


# COCO class IDs used by the pretrained YOLO model:
# 0 = person, 32 = sports ball
TARGET_CLASSES = {
    0: "player1",
    32: "ball",
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
LABELME_VERSION = "5.4.1"
DEFAULT_MODEL = "yolov8x.pt"
DEFAULT_IMAGE_SIZE = 1280
DEFAULT_INFERENCE_CONFIDENCE = 0.15
PLAYER_CONFIDENCE = 0.25
BALL_CONFIDENCE = 0.15


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate LabelMe JSON pre-annotations for football match frames."
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Folder containing the football images to process.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"YOLO model weights (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help=f"YOLO inference image size (default: {DEFAULT_IMAGE_SIZE}).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_INFERENCE_CONFIDENCE,
        help=f"YOLO inference confidence floor (default: {DEFAULT_INFERENCE_CONFIDENCE}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing LabelMe JSON files. Without this flag, existing files are skipped.",
    )
    return parser.parse_args()


def build_labelme_annotation(image_path, model, image_size, inference_confidence):
    """Run YOLO on one image and return its LabelMe annotation payload."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("OpenCV could not read this image.")

    height, width = image.shape[:2]
    result = model.predict(
        source=str(image_path),
        imgsz=image_size,
        conf=inference_confidence,
        verbose=False,
    )[0]

    shapes = []
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())

        if class_id not in TARGET_CLASSES:
            continue

        # Apply separate confidence thresholds for people and the smaller ball.
        if class_id == 0 and confidence < PLAYER_CONFIDENCE:
            continue
        if class_id == 32 and confidence < BALL_CONFIDENCE:
            continue

        x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
        shapes.append(
            {
                "label": TARGET_CLASSES[class_id],
                "points": [
                    [float(x_min), float(y_min)],
                    [float(x_max), float(y_max)],
                ],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {},
            }
        )

    return {
        "version": LABELME_VERSION,
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }


def main():
    args = parse_args()
    image_dir = Path(args.image_dir).expanduser()

    if not image_dir.is_dir():
        raise SystemExit(f"Error: image directory does not exist: {image_dir}")

    image_paths = sorted(
        (
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )

    if not image_paths:
        raise SystemExit(f"No supported images found in: {image_dir}")

    print(f"Found {len(image_paths)} images in: {image_dir}")
    print(f"Loading YOLO model: {args.model}")
    model = YOLO(args.model)

    generated = 0
    skipped = 0
    failed = 0

    for image_path in image_paths:
        json_path = image_path.with_suffix(".json")

        if json_path.exists() and not args.overwrite:
            print(f"[SKIP] Annotation already exists: {json_path.name}")
            skipped += 1
            continue

        try:
            annotation = build_labelme_annotation(
                image_path=image_path,
                model=model,
                image_size=args.imgsz,
                inference_confidence=args.conf,
            )

            with json_path.open("w", encoding="utf-8") as annotation_file:
                json.dump(annotation, annotation_file, ensure_ascii=False, indent=2)

            print(
                f"[DONE] {json_path.name} | "
                f"Generated {len(annotation['shapes'])} shapes"
            )
            generated += 1

        except Exception as error:
            print(f"[ERROR] {image_path.name}: {error}")
            failed += 1

    print("\nProcessing complete.")
    print(f"Generated: {generated}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")
    print("\nOpen the image folder in LabelMe to review the generated annotations.")
    print("Remember: YOLO labels are preliminary and require human verification.")


if __name__ == "__main__":
    main()
