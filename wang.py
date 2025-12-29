from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from utils import cv_centercrop, cv_randomcrops


def wang_features(img: np.ndarray) -> torch.Tensor:
    """Feature extraction from Wang et al.

    Reference: https://www.sciencedirect.com/science/article/pii/S1742287617300439
    """
    k = np.asarray([[0.0, 0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0]])

    filt_y = cv2.filter2D(img, ddepth=-1, kernel=k)
    filt_x = cv2.filter2D(img, ddepth=-1, kernel=k.T)
    res_x = (img - filt_x)[1:-1, 1:-1]
    res_y = (img - filt_y)[1:-1, 1:-1]

    feats = []
    for res in [res_x, res_y]:
        unfold = torch.nn.Unfold(kernel_size=5, dilation=1, padding=0, stride=1)
        res_tensor = torch.tensor(res, dtype=torch.float).unsqueeze(0).unsqueeze(1)
        patches = unfold(res_tensor)[0]

        patches = patches.reshape((5, 5, -1))
        corr_coff = torch.zeros((5, 5))
        p_center = patches[2, 2, :]

        for i in range(5):
            for j in range(5):
                p_ij = patches[i, j, :]
                corr_coff[i, j] = torch.corrcoef(torch.stack((p_center, p_ij), dim=0))[0, 1]

        feat = torch.cat(
            (
                corr_coff[0, :],
                corr_coff[1, 1:],
                corr_coff[2, 3:],
                corr_coff[3, 3:],
                corr_coff[4, 4].unsqueeze(0),
            )
        )
        feats.append(feat)

    return torch.cat(feats)


def _load_annots(data_path: Path) -> list[dict]:
    with (data_path / "annots.json").open() as handle:
        return json.load(handle)


def _load_grayscale(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def wang(data_path: Path, crop_center: bool = False) -> None:
    annots = _load_annots(data_path)
    new_annots = []
    for annot in tqdm(annots):
        path = data_path / annot["path"]
        x, y, w, h = annot["box"]

        img = _load_grayscale(path)
        img = img[int(y) : int(y + h), int(x) : int(x + w)]
        if crop_center:
            img = cv_centercrop(img)

        feat = wang_features(img)
        new_annots.append(
            {
                "id": f"{annot['id']:06d}",
                "feat": feat.numpy().tolist(),
                "label": annot["label"],
            }
        )

    filename = "wang_center.json" if crop_center else "wang.json"
    with Path(filename).open("w") as handle:
        json.dump(new_annots, handle)


def wang_multiple(data_path: Path, crops: int = 5) -> None:
    annots = _load_annots(data_path)
    new_annots = []
    for annot in tqdm(annots):
        path = data_path / annot["path"]
        x, y, w, h = annot["box"]

        img = _load_grayscale(path)
        img = img[int(y) : int(y + h), int(x) : int(x + w)]
        imgs = cv_randomcrops(img, size=(224, 224), num=crops)

        for j, crop in enumerate(imgs):
            feat = wang_features(crop)
            new_annots.append(
                {
                    "id": f"{annot['id']:06d}_{j}",
                    "feat": feat.numpy().tolist(),
                    "label": annot["label"],
                }
            )

    with Path("wang_multiple.json").open("w") as handle:
        json.dump(new_annots, handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Wang et al features.")
    parser.add_argument(
        "mode",
        choices=["center", "standard", "multiple"],
        help="Feature extraction mode.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data_dlc"),
        help="Path to the DLC2021 dataset root or a symlink named data_dlc.",
    )
    parser.add_argument("--crops", type=int, default=5, help="Number of crops for multiple mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve() if args.root.is_symlink() else args.root

    if args.mode == "center":
        print("Computing Wang et al features with center crop")
        wang(root, crop_center=True)
    elif args.mode == "standard":
        print("Computing Wang et al features with full document")
        wang(root, crop_center=False)
    elif args.mode == "multiple":
        print("Computing Wang et al features with 5 crops per image")
        wang_multiple(root, crops=args.crops)


if __name__ == "__main__":
    main()
