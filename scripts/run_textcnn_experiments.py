from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from news_alpha.data import load_daily_news, with_target
from news_alpha.light_filter import fit_lightweight_filter
from news_alpha.splits import build_indices
from news_alpha.textcnn import TrainConfig, train_and_predict_infer, train_and_predict_proba


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    out = {
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    out["roc_auc"] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    return out


def build_plain_text_variant(df, variant: str) -> list[str]:
    if variant == "daily_text":
        return df["daily_text"].tolist()
    if variant == "top3_concat":
        return [" ".join(headlines[:3]) for headlines in df["headlines"].tolist()]
    if variant == "top5_concat":
        return [" ".join(headlines[:5]) for headlines in df["headlines"].tolist()]
    raise ValueError(f"Unsupported variant: {variant}")


def _build_filtered_text_variant(
    df,
    fit_indices: np.ndarray,
    top_k: int,
    keyword_weight: float,
    tfidf_weight: float,
    filter_min_df: int,
    filter_max_features: int,
) -> list[str]:
    train_headlines: list[str] = []
    for i in fit_indices:
        train_headlines.extend(df.iloc[int(i)]["headlines"])

    rel_filter = fit_lightweight_filter(
        train_headlines=train_headlines,
        keyword_weight=keyword_weight,
        tfidf_weight=tfidf_weight,
        min_df=filter_min_df,
        max_features=filter_max_features,
    )

    texts = []
    for headlines in df["headlines"].tolist():
        selected = rel_filter.select_top(headlines, top_k=top_k)
        texts.append(" ".join(selected))
    return texts


def prepare_variant_texts(
    df,
    variant: str,
    splits: dict[str, object],
    keyword_weight: float,
    tfidf_weight: float,
    filter_min_df: int,
    filter_max_features: int,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    fold_texts: dict[int, np.ndarray] = {}

    for fold_idx, fold in enumerate(splits["cv_folds"], start=1):
        if variant in {"daily_text", "top3_concat", "top5_concat"}:
            texts = build_plain_text_variant(df, variant)
        elif variant == "filtered_top3":
            texts = _build_filtered_text_variant(
                df=df,
                fit_indices=fold["train"],
                top_k=3,
                keyword_weight=keyword_weight,
                tfidf_weight=tfidf_weight,
                filter_min_df=filter_min_df,
                filter_max_features=filter_max_features,
            )
        elif variant == "filtered_top5":
            texts = _build_filtered_text_variant(
                df=df,
                fit_indices=fold["train"],
                top_k=5,
                keyword_weight=keyword_weight,
                tfidf_weight=tfidf_weight,
                filter_min_df=filter_min_df,
                filter_max_features=filter_max_features,
            )
        else:
            raise ValueError(f"Unsupported variant: {variant}")

        fold_texts[fold_idx] = np.asarray(texts, dtype=object)

    if variant in {"daily_text", "top3_concat", "top5_concat"}:
        holdout_texts = build_plain_text_variant(df, variant)
    elif variant == "filtered_top3":
        holdout_texts = _build_filtered_text_variant(
            df=df,
            fit_indices=splits["train_val"],
            top_k=3,
            keyword_weight=keyword_weight,
            tfidf_weight=tfidf_weight,
            filter_min_df=filter_min_df,
            filter_max_features=filter_max_features,
        )
    elif variant == "filtered_top5":
        holdout_texts = _build_filtered_text_variant(
            df=df,
            fit_indices=splits["train_val"],
            top_k=5,
            keyword_weight=keyword_weight,
            tfidf_weight=tfidf_weight,
            filter_min_df=filter_min_df,
            filter_max_features=filter_max_features,
        )
    else:
        raise ValueError(f"Unsupported variant: {variant}")

    return fold_texts, np.asarray(holdout_texts, dtype=object)


def evaluate_config_for_variant(
    fold_texts: dict[int, np.ndarray],
    y: np.ndarray,
    splits: dict[str, object],
    config: TrainConfig,
    device: str | None,
) -> dict[str, object]:
    fold_metrics = []

    for fold_idx, fold in enumerate(splits["cv_folds"], start=1):
        text_arr = fold_texts[fold_idx]
        tr = fold["train"]
        va = fold["val"]
        probs = train_and_predict_proba(
            train_texts=text_arr[tr].tolist(),
            train_y=y[tr],
            val_texts=text_arr[va].tolist(),
            val_y=y[va],
            config=config,
            device=device,
        )
        metrics = evaluate_predictions(y[va], probs)
        metrics["fold"] = fold_idx
        fold_metrics.append(metrics)

    means = {
        k: float(np.mean([m[k] for m in fold_metrics]))
        for k in ("mcc", "f1", "accuracy", "roc_auc")
    }
    return {"cv_fold_metrics": fold_metrics, "cv_mean_metrics": means}


def fit_holdout_for_variant(
    holdout_text_arr: np.ndarray,
    y: np.ndarray,
    splits: dict[str, object],
    config: TrainConfig,
    device: str | None,
) -> dict[str, float]:
    train_val_idx = splits["train_val"]
    holdout_idx = splits["test"]

    tv_texts = holdout_text_arr[train_val_idx].tolist()
    tv_y = y[train_val_idx]
    cut = int(len(tv_texts) * 0.9)
    if cut < 32:
        cut = max(1, len(tv_texts) - 1)

    train_texts = tv_texts[:cut]
    train_y = tv_y[:cut]
    early_texts = tv_texts[cut:]
    early_y = tv_y[cut:]

    if len(early_texts) == 0:
        early_texts = train_texts[-min(64, len(train_texts)) :]
        early_y = train_y[-min(64, len(train_y)) :]

    holdout_probs = train_and_predict_infer(
        train_texts=train_texts,
        train_y=train_y,
        early_stop_texts=early_texts,
        early_stop_y=early_y,
        infer_texts=holdout_text_arr[holdout_idx].tolist(),
        config=config,
        device=device,
    )
    return evaluate_predictions(y[holdout_idx], holdout_probs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fast TextCNN experiments on news text.")
    parser.add_argument("--raw-csv", type=Path, default=ROOT / "Combined_News_DJIA.csv")
    parser.add_argument("--clean-csv", type=Path, default=ROOT / "Combined_News_DJIA_clean.csv")
    parser.add_argument("--target-mode", choices=("same_day", "next_day"), default="same_day")
    parser.add_argument("--keyword-weight", type=float, default=0.4)
    parser.add_argument("--tfidf-weight", type=float, default=0.6)
    parser.add_argument("--filter-min-df", type=int, default=2)
    parser.add_argument("--filter-max-features", type=int, default=30000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "outputs" / "textcnn_results.json",
    )
    args = parser.parse_args()

    df = with_target(load_daily_news(args.raw_csv, args.clean_csv), args.target_mode)
    splits = build_indices(df)
    y = df["Target"].to_numpy(dtype=int)

    variants = ("daily_text", "top3_concat", "top5_concat", "filtered_top3", "filtered_top5")
    configs = [
        TrainConfig(max_len=96, embed_dim=128, num_filters=96, epochs=6, batch_size=64, seed=42),
        TrainConfig(max_len=128, embed_dim=128, num_filters=128, epochs=8, batch_size=64, seed=42),
    ]

    experiments: dict[str, dict[str, object]] = {}
    for variant in variants:
        fold_texts, holdout_text_arr = prepare_variant_texts(
            df=df,
            variant=variant,
            splits=splits,
            keyword_weight=args.keyword_weight,
            tfidf_weight=args.tfidf_weight,
            filter_min_df=args.filter_min_df,
            filter_max_features=args.filter_max_features,
        )
        runs = []
        for i, cfg in enumerate(configs, start=1):
            result = evaluate_config_for_variant(
                fold_texts=fold_texts,
                y=y,
                splits=splits,
                config=cfg,
                device=args.device,
            )
            runs.append(
                {
                    "config_id": i,
                    "config": cfg.__dict__,
                    "cv_fold_metrics": result["cv_fold_metrics"],
                    "cv_mean_metrics": result["cv_mean_metrics"],
                }
            )

        best = sorted(
            runs,
            key=lambda r: (
                r["cv_mean_metrics"]["mcc"],
                r["cv_mean_metrics"]["f1"],
                r["cv_mean_metrics"]["accuracy"],
            ),
            reverse=True,
        )[0]

        best_cfg = TrainConfig(**best["config"])
        holdout = fit_holdout_for_variant(
            holdout_text_arr=holdout_text_arr,
            y=y,
            splits=splits,
            config=best_cfg,
            device=args.device,
        )
        experiments[variant] = {
            "best_config": best["config"],
            "cv_best_metrics": best["cv_mean_metrics"],
            "cv_best_fold_metrics": best["cv_fold_metrics"],
            "holdout_metrics": holdout,
        }

    ranking = sorted(
        (
            {
                "variant": name,
                "cv_mcc": r["cv_best_metrics"]["mcc"],
                "cv_f1": r["cv_best_metrics"]["f1"],
                "cv_accuracy": r["cv_best_metrics"]["accuracy"],
                "holdout_mcc": r["holdout_metrics"]["mcc"],
                "holdout_f1": r["holdout_metrics"]["f1"],
                "holdout_accuracy": r["holdout_metrics"]["accuracy"],
            }
            for name, r in experiments.items()
        ),
        key=lambda x: (x["cv_mcc"], x["cv_f1"], x["cv_accuracy"]),
        reverse=True,
    )

    payload = {
        "target_mode": args.target_mode,
        "dataset_rows": len(df),
        "train_val_range": splits["train_val_range"],
        "test_range": splits["test_range"],
        "filter_settings": {
            "keyword_weight": args.keyword_weight,
            "tfidf_weight": args.tfidf_weight,
            "min_df": args.filter_min_df,
            "max_features": args.filter_max_features,
        },
        "experiments": experiments,
        "ranking_by_cv": ranking,
        "best_variant": ranking[0] if ranking else None,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved TextCNN results to: {args.output_json}")
    if ranking:
        print("Top variant by CV:", ranking[0]["variant"])
        print("Top holdout metrics:", experiments[ranking[0]["variant"]]["holdout_metrics"])


if __name__ == "__main__":
    main()
