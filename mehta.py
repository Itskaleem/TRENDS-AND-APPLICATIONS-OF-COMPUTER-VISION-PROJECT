from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn.functional import conv2d
from tqdm import tqdm

from utils import cv_centercrop, crop_divisible


def _mehta_filters() -> torch.Tensor:
    root_2 = math.sqrt(2.0)
    weights = torch.stack(
        [
            torch.tensor([[1.0, 1.0], [-1.0, -1.0]]),
            torch.tensor([[1.0, -1.0], [1.0, -1.0]]),
            torch.tensor([[0.0, root_2], [-root_2, 0.0]]),
            torch.tensor([[root_2, 0.0], [0.0, -root_2]]),
            torch.tensor([[2.0, -2.0], [-2.0, 2.0]]),
        ],
        dim=0,
    )
    return weights.view(5, 1, 2, 2).to(torch.float)


def _extract_descriptor(
    img: np.ndarray,
    weights: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    size_h, size_w = img.shape[0] // 4, img.shape[1] // 4
    unfold = torch.nn.Unfold(kernel_size=(size_h, size_w), dilation=1, padding=0, stride=(size_h, size_w))
    img_tensor = torch.tensor(img.astype(float) / 255.0, dtype=torch.float).unsqueeze(0).unsqueeze(1)
    patches = unfold(img_tensor)[0].reshape((-1, 1, size_h, size_w))

    output = conv2d(patches, weights, bias=None, stride=2)
    bins = output.reshape(16, 5, -1)
    desc = torch.zeros(16, 5)

    for j in range(16):
        idxs = torch.zeros(5).to(torch.long)
        maxs, argmaxs = torch.max(bins[j], dim=0)
        idx, max_b = torch.unique(argmaxs[maxs > threshold], return_counts=True)
        idxs[idx] = max_b
        desc[j] = idxs

    desc_loc = torch.mean(desc, dim=0)
    desc_glob = torch.sum(desc, dim=0)
    desc_all = torch.sum(desc)
    return torch.cat((desc.flatten(), desc_loc, desc_glob, desc_all.unsqueeze(0)), dim=0)


def all_near(
    root: Path,
    aliasing_filter: bool = False,
    threshold: float = 1.0,
    crop_center: bool = False,
) -> None:
    with (root / "annots.json").open() as handle:
        annots = json.load(handle)

    weights = _mehta_filters()
    new_annots = []

    for annot in tqdm(annots):
        path = root / annot["path"]
        x, y, w, h = annot["box"]

        img = cv2.imread(str(path))
        img = img[int(y) : int(y + h), int(x) : int(x + w)]

        if crop_center:
            img = cv_centercrop(img)
        else:
            img = crop_divisible(img, divisor=8)

        if aliasing_filter:
            img = cv2.medianBlur(img, ksize=3)

        img = cv2.Canny(img, 100, 200)
        final_desc = _extract_descriptor(img, weights, threshold)

        new_annots.append(
            {
                "id": f"{annot['id']:06d}",
                "feat": final_desc.numpy().tolist(),
                "label": annot["label"],
            }
        )

    filename = "mehta_filter.json" if aliasing_filter else "mehta.json"
    with Path(filename).open("w") as handle:
        json.dump(new_annots, handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Mehta et al features.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data_dlc"),
        help="Path to the DLC2021 dataset root or a symlink named data_dlc.",
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="Apply anti-aliasing filter before edge detection.",
    )
    parser.add_argument("--threshold", type=float, default=1.0, help="Edge threshold.")
    parser.add_argument(
        "--center-crop",
        action="store_true",
        help="Center crop before processing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve() if args.root.is_symlink() else args.root
    message = "with anti-aliasing filter" if args.filter else "without anti-aliasing filter"
    print(f"Computing Mehta et al features {message}")
    all_near(root, aliasing_filter=args.filter, threshold=args.threshold, crop_center=args.center_crop)


if __name__ == "__main__":
    main()
