from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable


LOGGER = logging.getLogger(__name__)


def _load_split_annotations(path: Path) -> Dict[str, Dict]:
    with path.open() as handle:
        data = json.load(handle)
    return data["_via_img_metadata"]


def _iter_images(path: Path) -> Iterable[Path]:
    for img in sorted(path.iterdir()):
        if img.is_file():
            yield img


def make_annots(root: Path, remove_duplicates: bool = True) -> None:
    """Generate annots.json for the DLC2021 dataset layout."""
    annots = []
    annot_id = 0
    for capture_type in ["or", "re"]:
        images_root = root / capture_type / "images"
        annotations_root = root / capture_type / "annotations"

        for doc_type in sorted(images_root.iterdir()):
            for split in sorted((doc_type).iterdir()):
                img_path = split
                annot_path = annotations_root / doc_type.name / f"{split.name}.json"
                split_annot = _load_split_annotations(annot_path)

                for img in _iter_images(img_path):
                    if "(" in img.name:
                        if remove_duplicates:
                            LOGGER.warning("Removing duplicate image %s", img)
                            img.unlink(missing_ok=True)
                        continue

                    xs = ys = None
                    for metadata in split_annot.values():
                        if metadata["filename"] == img.name:
                            attrs = metadata["regions"][0]["shape_attributes"]
                            xs = attrs["all_points_x"]
                            ys = attrs["all_points_y"]
                            break

                    if xs is None or ys is None:
                        raise RuntimeError(f"image {img.name} not found in {annot_path}")

                    min_w = max(0, min(xs))
                    min_h = max(0, min(ys))
                    width = max(xs) - min_w
                    height = max(ys) - min_h
                    annots.append(
                        {
                            "id": annot_id,
                            "path": str(Path(capture_type) / "images" / doc_type.name / split.name / img.name),
                            "box": [min_w, min_h, width, height],
                            "label": 1 if capture_type == "re" else 0,
                        }
                    )
                    annot_id += 1

    output_path = root / "annots.json"
    with output_path.open("w") as handle:
        json.dump(annots, handle)
    LOGGER.info("Wrote %s with %d annotations", output_path, len(annots))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate annots.json for DLC2021.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data_dlc"),
        help="Path to the DLC2021 dataset root or a symlink named data_dlc.",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep duplicate images instead of removing them.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    root = args.root
    if root.is_symlink():
        root = root.resolve()
    make_annots(root, remove_duplicates=not args.keep_duplicates)


if __name__ == "__main__":
    main()
