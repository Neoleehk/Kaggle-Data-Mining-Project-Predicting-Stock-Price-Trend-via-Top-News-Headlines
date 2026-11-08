from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from news_alpha.data import load_daily_news
from news_alpha.embeddings import (
    DEFAULT_FINBERT_MODEL,
    DEFAULT_SBERT_MODEL,
    build_embedding_cache,
    collect_unique_texts,
    save_embedding_cache,
)
from news_alpha.relevance import DEFAULT_ANCHORS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and save FinBERT/SBERT embedding cache for news texts."
    )
    parser.add_argument("--raw-csv", type=Path, default=ROOT / "Combined_News_DJIA.csv")
    parser.add_argument("--clean-csv", type=Path, default=ROOT / "Combined_News_DJIA_clean.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "embedding_cache",
    )
    parser.add_argument("--finbert-model", type=str, default=DEFAULT_FINBERT_MODEL)
    parser.add_argument("--sbert-model", type=str, default=DEFAULT_SBERT_MODEL)
    parser.add_argument("--finbert-batch-size", type=int, default=64)
    parser.add_argument("--sbert-batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument(
        "--strategy",
        choices=("full", "two_stage"),
        default="two_stage",
        help="two_stage computes FinBERT only on SBERT-selected top headlines plus anchors and daily text.",
    )
    parser.add_argument(
        "--finbert-top-n",
        type=int,
        default=5,
        help="Only used in two_stage strategy: FinBERT shortlist size per day from SBERT ranking.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--max-texts",
        type=int,
        default=None,
        help="Optional cap for quick dry-runs.",
    )
    args = parser.parse_args()

    df = load_daily_news(args.raw_csv, args.clean_csv)
    texts = collect_unique_texts(df, anchors=DEFAULT_ANCHORS)
    if args.max_texts is not None and args.max_texts > 0:
        anchors = [a for a in DEFAULT_ANCHORS if a in texts]
        non_anchors = [t for t in texts if t not in anchors]
        texts = anchors + non_anchors[: max(0, args.max_texts - len(anchors))]

    cache = build_embedding_cache(
        df=df,
        texts=texts,
        anchors=DEFAULT_ANCHORS,
        strategy=args.strategy,
        finbert_top_n=args.finbert_top_n,
        finbert_model=args.finbert_model,
        sbert_model=args.sbert_model,
        finbert_batch_size=args.finbert_batch_size,
        sbert_batch_size=args.sbert_batch_size,
        max_length=args.max_length,
        device=args.device,
    )
    save_embedding_cache(cache, args.output_dir)
    print(f"Saved embedding cache to: {args.output_dir}")
    print(
        "Rows(sbert/all): "
        f"{len(cache.texts)} | Rows(finbert): {len(cache.finbert_texts)} "
        f"| FinBERT dim: {cache.finbert.shape[1]} | SBERT dim: {cache.sbert.shape[1]}"
    )


if __name__ == "__main__":
    main()
