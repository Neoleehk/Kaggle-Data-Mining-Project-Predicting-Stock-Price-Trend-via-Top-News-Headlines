from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from news_alpha.data import load_daily_news, with_target
from news_alpha.embeddings import load_embedding_cache
from news_alpha.features import build_feature_matrix
from news_alpha.splits import build_indices


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    out = {
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    out["roc_auc"] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    return out


def _build_model(c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=c_value,
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate_variant(
    features: np.ndarray,
    target: np.ndarray,
    splits: dict[str, object],
    c_grid: list[float],
) -> dict[str, object]:
    model_runs: list[dict[str, object]] = []

    for c_value in c_grid:
        fold_metrics = []
        for fold_idx, fold in enumerate(splits["cv_folds"], start=1):
            model = _build_model(c_value)
            train_idx = fold["train"]
            val_idx = fold["val"]
            model.fit(features[train_idx], target[train_idx])
            probs = model.predict_proba(features[val_idx])[:, 1]
            metrics = evaluate_predictions(target[val_idx], probs)
            metrics["fold"] = fold_idx
            fold_metrics.append(metrics)

        mean_metrics = {
            metric: float(np.mean([m[metric] for m in fold_metrics]))
            for metric in ("mcc", "f1", "accuracy", "roc_auc")
        }
        model_runs.append(
            {
                "C": c_value,
                "cv_fold_metrics": fold_metrics,
                "cv_mean_metrics": mean_metrics,
            }
        )

    best = sorted(
        model_runs,
        key=lambda r: (
            r["cv_mean_metrics"]["mcc"],
            r["cv_mean_metrics"]["f1"],
            r["cv_mean_metrics"]["accuracy"],
        ),
        reverse=True,
    )[0]

    final_model = _build_model(float(best["C"]))
    final_model.fit(features[splits["train_val"]], target[splits["train_val"]])
    holdout_prob = final_model.predict_proba(features[splits["test"]])[:, 1]
    holdout = evaluate_predictions(target[splits["test"]], holdout_prob)

    return {
        "best_C": best["C"],
        "cv_best_metrics": best["cv_mean_metrics"],
        "cv_best_fold_metrics": best["cv_fold_metrics"],
        "holdout_metrics": holdout,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run embedding-based feature experiments.")
    parser.add_argument("--raw-csv", type=Path, default=ROOT / "Combined_News_DJIA.csv")
    parser.add_argument("--clean-csv", type=Path, default=ROOT / "Combined_News_DJIA_clean.csv")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "outputs" / "embedding_cache",
    )
    parser.add_argument("--target-mode", choices=("same_day", "next_day"), default="same_day")
    parser.add_argument(
        "--finbert-top-n",
        type=int,
        default=5,
        help="Top-N headlines per day to fuse with FinBERT score; others use SBERT-only relevance.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "outputs" / "embedding_experiment_results.json",
    )
    args = parser.parse_args()

    df = with_target(load_daily_news(args.raw_csv, args.clean_csv), args.target_mode)
    splits = build_indices(df)
    cache = load_embedding_cache(args.cache_dir)
    y = df["Target"].to_numpy(dtype=int)

    c_grid = [0.1, 1.0, 10.0]
    relevance_weights = [0.3, 0.5, 0.7]
    ewma_alphas = [0.2, 0.4, 0.6, 0.8]

    experiments: dict[str, dict[str, object]] = {}

    x_f0 = build_feature_matrix(df, cache, variant="f0_daily_text", finbert_top_n=args.finbert_top_n)
    experiments["f0_daily_text"] = evaluate_variant(x_f0, y, splits, c_grid)

    for w in relevance_weights:
        for variant in ("f1_top1", "f2_top3", "f3_top5", "f4_all_weighted"):
            key = f"{variant}_w{w:.1f}"
            features = build_feature_matrix(
                df,
                cache,
                variant=variant,
                relevance_weight=w,
                finbert_top_n=args.finbert_top_n,
            )
            experiments[key] = evaluate_variant(features, y, splits, c_grid)

        for alpha in ewma_alphas:
            key = f"f5_ewma_top3_w{w:.1f}_a{alpha:.1f}"
            features = build_feature_matrix(
                df,
                cache,
                variant="f5_ewma_top3",
                relevance_weight=w,
                alpha=alpha,
                finbert_top_n=args.finbert_top_n,
            )
            experiments[key] = evaluate_variant(features, y, splits, c_grid)

    ranked = sorted(
        (
            {
                "experiment": name,
                "cv_mcc": result["cv_best_metrics"]["mcc"],
                "cv_f1": result["cv_best_metrics"]["f1"],
                "cv_accuracy": result["cv_best_metrics"]["accuracy"],
                "holdout_mcc": result["holdout_metrics"]["mcc"],
                "holdout_f1": result["holdout_metrics"]["f1"],
                "holdout_accuracy": result["holdout_metrics"]["accuracy"],
            }
            for name, result in experiments.items()
        ),
        key=lambda r: (r["cv_mcc"], r["cv_f1"], r["cv_accuracy"]),
        reverse=True,
    )

    payload = {
        "target_mode": args.target_mode,
        "dataset_rows": len(df),
        "train_val_range": splits["train_val_range"],
        "test_range": splits["test_range"],
        "cache_strategy": cache.strategy,
        "cache_finbert_rows": len(cache.finbert_texts),
        "cache_sbert_rows": len(cache.texts),
        "finbert_top_n_runtime": args.finbert_top_n,
        "experiments": experiments,
        "ranking_by_cv": ranked,
        "best_experiment": ranked[0] if ranked else None,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved embedding experiment results to: {args.output_json}")
    if ranked:
        print("Top experiment by CV:", ranked[0]["experiment"])
        print("Top holdout metrics:", experiments[ranked[0]["experiment"]]["holdout_metrics"])


if __name__ == "__main__":
    main()
