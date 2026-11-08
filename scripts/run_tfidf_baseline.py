from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from news_alpha.baseline import candidate_models, fit_and_test_best, run_cv
from news_alpha.data import load_daily_news, with_target
from news_alpha.splits import build_indices


def _date_range(df: pd.DataFrame, idx) -> tuple[str, str]:
    subset = df.iloc[idx]
    return (
        subset["Date"].iloc[0].strftime("%Y-%m-%d"),
        subset["Date"].iloc[-1].strftime("%Y-%m-%d"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run baseline TF-IDF experiments on daily cleaned news text."
    )
    parser.add_argument(
        "--raw-csv",
        type=Path,
        default=ROOT / "Combined_News_DJIA.csv",
    )
    parser.add_argument(
        "--clean-csv",
        type=Path,
        default=ROOT / "Combined_News_DJIA_clean.csv",
    )
    parser.add_argument(
        "--target-mode",
        choices=("same_day", "next_day"),
        default="same_day",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "outputs" / "baseline_results.json",
    )
    parser.add_argument(
        "--include-svm",
        action="store_true",
        help="Include linear SVM candidates (slower).",
    )
    args = parser.parse_args()

    df = load_daily_news(args.raw_csv, args.clean_csv)
    df = with_target(df, args.target_mode)

    splits = build_indices(df)
    texts = df["daily_text"].to_numpy(dtype=object)
    targets = df["Target"].to_numpy(dtype=int)

    configs = candidate_models(include_svm=args.include_svm)
    cv_results = run_cv(
        texts=texts,
        targets=targets,
        folds=splits["cv_folds"],
        configs=configs,
    )

    best = sorted(
        cv_results,
        key=lambda r: (
            r["cv_mean_metrics"]["mcc"],
            r["cv_mean_metrics"]["f1"],
            r["cv_mean_metrics"]["accuracy"],
        ),
        reverse=True,
    )[0]

    holdout_metrics = fit_and_test_best(
        texts=texts,
        targets=targets,
        train_val_idx=splits["train_val"],
        test_idx=splits["test"],
        best=best,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "target_mode": args.target_mode,
        "dataset_rows": len(df),
        "train_val_range": splits["train_val_range"],
        "test_range": splits["test_range"],
        "train_val_dates": _date_range(df, splits["train_val"]),
        "test_dates": _date_range(df, splits["test"]),
        "best_model": best,
        "holdout_metrics": holdout_metrics,
    }
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved baseline results to: {args.output_json}")
    print(f"Best CV model: {best['model_name']}")
    print("CV mean metrics:", best["cv_mean_metrics"])
    print("Holdout metrics:", holdout_metrics)


if __name__ == "__main__":
    main()
