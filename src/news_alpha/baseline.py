from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


@dataclass(frozen=True)
class CandidateModel:
    name: str
    classifier: str
    c: float
    ngram: tuple[int, int]
    min_df: int
    max_features: int


def candidate_models(include_svm: bool = False) -> list[CandidateModel]:
    configs: list[CandidateModel] = []

    for c, ngram, min_df, max_features in product(
        (0.1, 1.0, 10.0),
        ((1, 1), (1, 2)),
        (2, 5),
        (15000, 30000),
    ):
        configs.append(
            CandidateModel(
                name=f"logreg_c{c}_ng{ngram[0]}{ngram[1]}_mindf{min_df}_maxf{max_features}",
                classifier="logreg",
                c=c,
                ngram=ngram,
                min_df=min_df,
                max_features=max_features,
            )
        )
        if include_svm:
            configs.append(
                CandidateModel(
                    name=f"svm_c{c}_ng{ngram[0]}{ngram[1]}_mindf{min_df}_maxf{max_features}",
                    classifier="svm",
                    c=c,
                    ngram=ngram,
                    min_df=min_df,
                    max_features=max_features,
                )
            )

    return configs


def _build_pipeline(config: CandidateModel) -> Pipeline:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=config.ngram,
        min_df=config.min_df,
        max_features=config.max_features,
    )

    if config.classifier == "logreg":
        clf = LogisticRegression(
            C=config.c,
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )
    elif config.classifier == "svm":
        clf = SVC(
            C=config.c,
            kernel="linear",
            class_weight="balanced",
            probability=True,
            random_state=42,
        )
    else:
        raise ValueError(f"Unsupported classifier: {config.classifier}")

    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def run_cv(
    texts: np.ndarray,
    targets: np.ndarray,
    folds: Iterable[dict[str, object]],
    configs: list[CandidateModel],
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for config in configs:
        fold_metrics = []
        for fold_idx, fold in enumerate(folds, start=1):
            train_idx = fold["train"]
            val_idx = fold["val"]

            model = _build_pipeline(config)
            model.fit(texts[train_idx], targets[train_idx])
            y_prob = model.predict_proba(texts[val_idx])[:, 1]
            metrics = evaluate_predictions(targets[val_idx], y_prob)
            metrics["fold"] = fold_idx
            fold_metrics.append(metrics)

        means = {
            key: float(np.mean([m[key] for m in fold_metrics]))
            for key in ("mcc", "f1", "accuracy", "roc_auc")
        }
        results.append(
            {
                "model_name": config.name,
                "classifier": config.classifier,
                "params": {
                    "C": config.c,
                    "ngram": config.ngram,
                    "min_df": config.min_df,
                    "max_features": config.max_features,
                },
                "cv_fold_metrics": fold_metrics,
                "cv_mean_metrics": means,
            }
        )
    return results


def fit_and_test_best(
    texts: np.ndarray,
    targets: np.ndarray,
    train_val_idx: np.ndarray,
    test_idx: np.ndarray,
    best: dict[str, object],
) -> dict[str, float]:
    params = best["params"]
    config = CandidateModel(
        name=best["model_name"],
        classifier=best["classifier"],
        c=float(params["C"]),
        ngram=tuple(params["ngram"]),
        min_df=int(params["min_df"]),
        max_features=int(params["max_features"]),
    )

    model = _build_pipeline(config)
    model.fit(texts[train_val_idx], targets[train_val_idx])
    y_prob = model.predict_proba(texts[test_idx])[:, 1]
    return evaluate_predictions(targets[test_idx], y_prob)
