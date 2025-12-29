from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50
from tqdm import tqdm

from utils import cv_centercrop


def _load_annots(root: Path) -> list[dict]:
    with (root / "annots.json").open() as handle:
        return json.load(handle)


def _load_model(model_type: int) -> torch.nn.Module:
    if model_type == 18:
        print("Extracting feats from pretrained ResNet18")
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    elif model_type == 50:
        print("Extracting feats from pretrained ResNet50")
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
    else:
        raise RuntimeError(f"Model ResNet-{model_type} not recognized.")

    model.fc = torch.nn.Identity()
    model.eval()
    return model


def resnet_feats(root: Path, model_type: int = 18) -> None:
    model = _load_model(model_type)
    annots = _load_annots(root)

    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    new_annots = []
    for annot in tqdm(annots):
        path = root / annot["path"]
        x, y, w, h = annot["box"]

        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img[int(y) : int(y + h), int(x) : int(x + w)]
        img = cv_centercrop(img)

        img_tensor = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        img_tensor = (img_tensor - mean.view(3, 1, 1)) / std.view(3, 1, 1)
        img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            res = model(img_tensor)

        new_annots.append(
            {
                "id": f"{annot['id']:06d}",
                "feat": res.squeeze(0).numpy().tolist(),
                "label": annot["label"],
            }
        )

    with Path(f"resnet{model_type}_center.json").open("w") as handle:
        json.dump(new_annots, handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute ResNet features.")
    parser.add_argument("model", type=int, choices=[18, 50], help="ResNet model size.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data_dlc"),
        help="Path to the DLC2021 dataset root or a symlink named data_dlc.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve() if args.root.is_symlink() else args.root
    print("Computing ResNet features with center crop")
    resnet_feats(root, model_type=args.model)


if __name__ == "__main__":
    main()
