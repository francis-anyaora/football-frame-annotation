"""Split image files into numbered batches.

By default, images are MOVED from the source folder into batch folders.
"""

import argparse
import re
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def natural_sort_key(path: Path):
    """Sort filenames naturally (for example, frame2 before frame10)."""
    parts = re.split(r"(\d+)", path.name)
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part.casefold())
        for part in parts
    )


def split_images(
    source_folder: Path,
    destination_parent_folder: Path,
    batch_size: int = 200,
) -> None:
    """Move images from source_folder into numbered batch folders."""
    source_folder = source_folder.expanduser().resolve()
    destination_parent_folder = destination_parent_folder.expanduser().resolve()

    if not source_folder.is_dir():
        raise ValueError(f"Source folder does not exist or is not a folder: {source_folder}")

    if batch_size < 1:
        raise ValueError("Batch size must be at least 1.")

    if source_folder == destination_parent_folder:
        raise ValueError("The source and destination folders cannot be the same.")

    if source_folder in destination_parent_folder.parents:
        raise ValueError(
            "The destination cannot be inside the source folder. "
            "Choose the source folder's parent or another separate folder."
        )

    if destination_parent_folder.exists() and not destination_parent_folder.is_dir():
        raise ValueError(
            f"Destination path exists but is not a folder: {destination_parent_folder}"
        )

    image_files = [
        path
        for path in source_folder.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    image_files.sort(key=natural_sort_key)

    total_images = len(image_files)
    print(f"Found {total_images} image(s) in: {source_folder}")

    if total_images == 0:
        print("No supported images found. Nothing to move.")
        return

    # Calculate all batch folders first and check for conflicts before moving files.
    batches = []
    for start_index in range(0, total_images, batch_size):
        batch_files = image_files[start_index : start_index + batch_size]
        first_number = start_index + 1
        last_number = start_index + len(batch_files)
        folder_name = f"batch_{first_number}_to_{last_number}"
        batch_folder = destination_parent_folder / folder_name

        if batch_folder.exists():
            raise FileExistsError(
                f"The batch folder already exists: {batch_folder}\n"
                "Rename or move that folder, or choose a different destination. "
                "No images have been moved."
            )

        batches.append((batch_files, batch_folder))

    destination_parent_folder.mkdir(parents=True, exist_ok=True)

    moved_count = 0
    for batch_files, batch_folder in batches:
        batch_folder.mkdir()

        for image_path in batch_files:
            destination_path = batch_folder / image_path.name
            shutil.move(str(image_path), str(destination_path))
            moved_count += 1

        print(f"Moved {len(batch_files)} image(s) to: {batch_folder.name}")

    print("-" * 60)
    print(f"Process complete. Moved {moved_count} image(s) into {len(batches)} batch folder(s).")
    print("The original image files are no longer in the source folder.")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Move image files into naturally sorted batches."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Folder containing the images to organize.",
    )
    parser.add_argument(
        "--destination",
        required=True,
        help="Parent folder in which the batch folders will be created.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Number of images per batch (default: 200).",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    try:
        split_images(
            source_folder=Path(args.source),
            destination_parent_folder=Path(args.destination),
            batch_size=args.batch_size,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
