"""Move images that are visually similar to a reference image into an output folder.

The script uses the CLIP image encoder provided by Sentence Transformers and
cosine similarity to compare each image against one reference image.
"""

import argparse
import shutil
from pathlib import Path
from typing import Optional

from PIL import Image
from sentence_transformers import SentenceTransformer, util

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MODEL_NAME = "clip-ViT-B-32"
DEFAULT_THRESHOLD = 0.88


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Find images visually similar to a reference image and move "
            "matching files into an output folder."
        )
    )
    parser.add_argument(
        "--main",
        required=True,
        type=Path,
        help="Folder containing the images to compare.",
    )
    parser.add_argument(
        "--guide",
        required=True,
        type=Path,
        help="Folder containing the reference image (unless --reference is supplied).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Folder where matching images will be moved.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help=(
            "Optional path to a specific reference image. When omitted, "
            "the first image in the guide folder alphabetically is used."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Minimum cosine similarity for a match (default: {DEFAULT_THRESHOLD}).",
    )
    return parser.parse_args()


def find_reference_image(guide_folder: Path, reference_path: Optional[Path]) -> Path:
    """Return the supplied reference image or the first image in the guide folder."""
    if reference_path is not None:
        if not reference_path.is_file():
            raise FileNotFoundError(f"Reference image not found: {reference_path}")
        if reference_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported reference image format: {reference_path.suffix}")
        return reference_path

    if not guide_folder.is_dir():
        raise NotADirectoryError(f"Guide folder does not exist: {guide_folder}")

    guide_images = sorted(
        path for path in guide_folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not guide_images:
        raise FileNotFoundError(f"No supported images found in guide folder: {guide_folder}")

    return guide_images[0]


def encode_image(model: SentenceTransformer, image_path: Path):
    """Load an image safely and return its CLIP embedding."""
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        return model.encode(rgb_image, convert_to_tensor=True)


def main():
    args = parse_arguments()

    main_folder = args.main.expanduser().resolve()
    guide_folder = args.guide.expanduser().resolve()
    output_folder = args.output.expanduser().resolve()
    reference_path = args.reference.expanduser().resolve() if args.reference else None

    if not main_folder.is_dir():
        raise NotADirectoryError(f"Main image folder does not exist: {main_folder}")

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("The similarity threshold must be between 0 and 1.")

    reference_image = find_reference_image(guide_folder, reference_path)

    # Do not allow the output folder to be the same folder being scanned.
    if output_folder == main_folder:
        raise ValueError("The output folder must be different from the main image folder.")

    image_files = sorted(
        path for path in main_folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_files:
        print(f"No supported images found in: {main_folder}")
        return

    print(f"Reference image: {reference_image.name}")
    print(f"Images to scan: {len(image_files)}")
    print(f"Similarity threshold: {args.threshold:.2f}")
    print(f"Loading CLIP model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)
    reference_embedding = encode_image(model, reference_image)

    output_folder.mkdir(parents=True, exist_ok=True)

    matched_count = 0
    skipped_count = 0
    error_count = 0

    for image_path in image_files:
        # Avoid moving the reference if it is also inside the main folder.
        if image_path.resolve() == reference_image.resolve():
            continue

        try:
            image_embedding = encode_image(model, image_path)
            similarity = util.cos_sim(reference_embedding, image_embedding).item()

            if similarity < args.threshold:
                continue

            destination_path = output_folder / image_path.name

            # Never overwrite an existing file in the output folder.
            if destination_path.exists():
                print(f"[SKIP] {image_path.name}: a file with that name already exists in the output folder.")
                skipped_count += 1
                continue

            shutil.move(str(image_path), str(destination_path))
            print(f"[MATCH] Moved {image_path.name} (similarity: {similarity:.3f})")
            matched_count += 1

        except Exception as error:
            print(f"[ERROR] Could not process {image_path.name}: {error}")
            error_count += 1

    print("\nProcessing complete.")
    print(f"Matched and moved: {matched_count}")
    print(f"Skipped because destination file exists: {skipped_count}")
    print(f"Errors: {error_count}")
    print(f"Output folder: {output_folder}")


if __name__ == "__main__":
    main()
