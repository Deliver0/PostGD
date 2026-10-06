"""Support-only few-shot head selection on frozen feature arrays.

The input NPZ must contain:
    support_features: [N, D]
    support_labels:   [N]
    query_features:   [M, D]
    query_labels:     [M]

Query labels are used only for the final report, never for candidate selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


DEFAULT_CANDIDATES = (
    "cosine_raw",
    "cosine_shrink_0.25",
    "cosine_shrink_0.4",
    "logreg_0.001",
    "logreg_0.01",
)


def normalize(features: np.ndarray) -> np.ndarray:
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)


def prototypes(features: np.ndarray, labels: np.ndarray, shrink: float = 0.0) -> np.ndarray:
    result = np.stack([features[labels == cls].mean(axis=0) for cls in (0, 1)])
    if shrink:
        result = (1.0 - shrink) * result + shrink * features.mean(axis=0, keepdims=True)
    return result


def cosine_scores(features: np.ndarray, class_prototypes: np.ndarray) -> np.ndarray:
    logits = normalize(features) @ normalize(class_prototypes).T
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
    return probabilities[:, 1]


def fit_scores(
    candidate: str,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    query_features: np.ndarray,
) -> np.ndarray:
    if candidate == "cosine_raw":
        return cosine_scores(query_features, prototypes(train_features, train_labels))
    if candidate.startswith("cosine_shrink_"):
        alpha = float(candidate.rsplit("_", 1)[1])
        return cosine_scores(query_features, prototypes(train_features, train_labels, alpha))
    if candidate.startswith("logreg_"):
        c_value = float(candidate.rsplit("_", 1)[1])
        classifier = LogisticRegression(
            C=c_value,
            solver="liblinear",
            max_iter=500,
            random_state=0,
        )
        classifier.fit(normalize(train_features), train_labels)
        return classifier.predict_proba(normalize(query_features))[:, 1]
    raise ValueError(f"unknown candidate: {candidate}")


def score(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
    }


def select_candidate(
    features: np.ndarray,
    labels: np.ndarray,
    candidates: tuple[str, ...],
    repeats: int,
) -> tuple[str, list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    for candidate in candidates:
        fold_scores: list[dict[str, float]] = []
        for repeat in range(repeats):
            splitter = StratifiedKFold(
                n_splits=2,
                shuffle=True,
                random_state=20261006 + repeat,
            )
            for train_idx, valid_idx in splitter.split(features, labels):
                probabilities = fit_scores(
                    candidate,
                    features[train_idx],
                    labels[train_idx],
                    features[valid_idx],
                )
                fold_scores.append(score(labels[valid_idx], probabilities))
        mean_auc = float(np.mean([item["auc"] for item in fold_scores]))
        mean_balanced = float(
            np.mean([item["balanced_accuracy"] for item in fold_scores])
        )
        summaries.append(
            {
                "candidate": candidate,
                "cv_auc": mean_auc,
                "cv_balanced_accuracy": mean_balanced,
                "selection_score": 0.5 * (mean_auc + mean_balanced),
                "folds": fold_scores,
            }
        )
    selected = max(
        summaries,
        key=lambda item: (item["selection_score"], item["cv_auc"]),
    )["candidate"]
    return selected, summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--support-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cv-repeats", type=int, default=5)
    args = parser.parse_args()

    arrays = np.load(args.support_npz)
    support_features = np.asarray(arrays["support_features"], dtype=np.float32)
    support_labels = np.asarray(arrays["support_labels"], dtype=np.int64)
    query_features = np.asarray(arrays["query_features"], dtype=np.float32)
    query_labels = np.asarray(arrays["query_labels"], dtype=np.int64)
    if support_features.ndim != 2 or query_features.ndim != 2:
        raise ValueError("feature arrays must have shape [N, D]")
    if support_features.shape[1] != query_features.shape[1]:
        raise ValueError("support and query feature dimensions differ")
    if set(np.unique(support_labels)) != {0, 1}:
        raise ValueError("support labels must contain both classes")

    selected, cv = select_candidate(
        support_features,
        support_labels,
        DEFAULT_CANDIDATES,
        args.cv_repeats,
    )
    probabilities = fit_scores(
        selected,
        support_features,
        support_labels,
        query_features,
    )
    result = {
        "selected_candidate": selected,
        "support_count": int(len(support_labels)),
        "query_count": int(len(query_labels)),
        "threshold": 0.5,
        "cv": cv,
        "query_metrics": score(query_labels, probabilities),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["query_metrics"], indent=2))


if __name__ == "__main__":
    main()

