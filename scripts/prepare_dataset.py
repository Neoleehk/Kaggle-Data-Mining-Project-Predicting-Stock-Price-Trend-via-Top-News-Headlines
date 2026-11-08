from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from news_alpha.data import load_daily_news
from news_alpha.splits import build_indices


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare cleaned daily dataset and split metadata."
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
        "--output-csv",
        type=Path,
        default=ROOT / "outputs" / "daily_news_prepared.csv",
    )
    parser.add_argument(
        "--output-splits-json",
        type=Path,
        default=ROOT / "outputs" / "split_metadata.json",
    )
    args = parser.parse_args()

    df = load_daily_news(args.raw_csv, args.clean_csv)
    splits = build_indices(df)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.assign(
        Date=df["Date"].dt.strftime("%Y-%m-%d"),
        headlines=df["headlines"].map(lambda x: " ||| ".join(x)),
    ).to_csv(args.output_csv, index=False)

    fold_payload = []
    for fold in splits["cv_folds"]:
        train_idx = fold["train"]
        val_idx = fold["val"]
        fold_payload.append(
            {
                "train_range": fold["train_range"],
                "val_range": fold["val_range"],
                "train_dates": [
                    df.iloc[train_idx]["Date"].iloc[0].strftime("%Y-%m-%d"),
                    df.iloc[train_idx]["Date"].iloc[-1].strftime("%Y-%m-%d"),
                ],
                "val_dates": [
                    df.iloc[val_idx]["Date"].iloc[0].strftime("%Y-%m-%d"),
                    df.iloc[val_idx]["Date"].iloc[-1].strftime("%Y-%m-%d"),
                ],
            }
        )

    payload = {
        "row_count": len(df),
        "date_min": df["Date"].iloc[0].strftime("%Y-%m-%d"),
        "date_max": df["Date"].iloc[-1].strftime("%Y-%m-%d"),
        "train_val_range": splits["train_val_range"],
        "test_range": splits["test_range"],
        "train_val_dates": [
            df.iloc[splits["train_val"]]["Date"].iloc[0].strftime("%Y-%m-%d"),
            df.iloc[splits["train_val"]]["Date"].iloc[-1].strftime("%Y-%m-%d"),
        ],
        "test_dates": [
            df.iloc[splits["test"]]["Date"].iloc[0].strftime("%Y-%m-%d"),
            df.iloc[splits["test"]]["Date"].iloc[-1].strftime("%Y-%m-%d"),
        ],
        "folds": fold_payload,
    }

    args.output_splits_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved prepared dataset to: {args.output_csv}")
    print(f"Saved split metadata to: {args.output_splits_json}")


if __name__ == "__main__":
    main()

