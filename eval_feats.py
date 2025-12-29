from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC


def _load_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    feats: list[list[float]] = []
    labels: list[int] = []
    with path.open() as handle:
        annots = json.load(handle)

    for annot in annots:
        feat = np.asarray(annot["feat"], dtype=np.float32)
        if np.any(np.isnan(feat)):
            continue
        feats.append(annot["feat"])
        labels.append(annot["label"])

    features = np.asarray(feats, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=np.float32)
    return features, labels_arr


def _normalize_features(features: np.ndarray) -> np.ndarray:
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    return (features - mean) / (std + 1e-6)


def eval_svm(
    data_path: Path,
    svm_iters: int = 100000,
    svm_conf: float = 1e-3,
    crossval_k: int = 10,
) -> Iterable[float]:
    """Evaluate features with a LinearSVM classifier."""
    feats, labels = _load_features(data_path)
    feats = _normalize_features(feats)
    clf = LinearSVC(max_iter=svm_iters, tol=svm_conf)
    return cross_val_score(clf, feats, labels, cv=crossval_k)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate features with a Linear SVM.")
    parser.add_argument("features", type=Path, help="Path to feature json file.")
    parser.add_argument("--svm-iters", type=int, default=100000, help="Max SVM iterations.")
    parser.add_argument("--svm-conf", type=float, default=1e-3, help="SVM tolerance.")
    parser.add_argument("--crossval-k", type=int, default=10, help="Cross-validation folds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.features.suffix != ".json":
        raise ValueError(f"Expected json file, found {args.features}")

    scores = eval_svm(
        args.features,
        svm_iters=args.svm_iters,
        svm_conf=args.svm_conf,
        crossval_k=args.crossval_k,
    )
    mean_acc, std_acc = np.mean(scores), np.std(scores)
    print(f"Evaluating on method {args.features}")
    print(f"{args.features.stem:<16} : {mean_acc:4.4f} {std_acc:1.4f}")


if __name__ == "__main__":
    main()
