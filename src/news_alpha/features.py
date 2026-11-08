from __future__ import annotations

import numpy as np
import pandas as pd

from .embeddings import EmbeddingCache
from .relevance import score_headline_indices, weighted_pool


def _empty_vector(cache: EmbeddingCache) -> np.ndarray:
    return np.zeros(cache.finbert.shape[1] + cache.sbert.shape[1], dtype=np.float32)


def _concat_embeddings(fin_vec: np.ndarray, sbert_vec: np.ndarray) -> np.ndarray:
    return np.concatenate([fin_vec, sbert_vec]).astype(np.float32)


def _headline_indices(headlines: list[str], text_to_index: dict[str, int]) -> np.ndarray:
    idx = [text_to_index[h] for h in headlines if h in text_to_index]
    return np.array(idx, dtype=int)


def _top_k_weighted_pool(
    indices: np.ndarray,
    scores: np.ndarray,
    top_k: int,
    cache: EmbeddingCache,
) -> np.ndarray:
    if indices.size == 0:
        return _empty_vector(cache)

    order = np.argsort(scores)[::-1]
    selected = order[: min(top_k, len(order))]
    selected_indices = indices[selected]
    selected_scores = scores[selected]
    return weighted_pool(selected_indices, selected_scores, cache)


def build_feature_matrix(
    df: pd.DataFrame,
    cache: EmbeddingCache,
    variant: str,
    relevance_weight: float = 0.5,
    alpha: float | None = None,
    finbert_top_n: int | None = None,
) -> np.ndarray:
    if variant == "f5_ewma_top3":
        if alpha is None or not (0 < alpha <= 1):
            raise ValueError("For f5_ewma_top3, alpha must be in (0, 1].")
        base = build_feature_matrix(
            df,
            cache,
            variant="f2_top3",
            relevance_weight=relevance_weight,
            finbert_top_n=finbert_top_n,
        )
        ewma = np.zeros_like(base)
        ewma[0] = base[0]
        for i in range(1, len(base)):
            ewma[i] = alpha * base[i] + (1 - alpha) * ewma[i - 1]
        return ewma

    rows: list[np.ndarray] = []

    for _, row in df.iterrows():
        if variant == "f0_daily_text":
            text = row["daily_text"]
            idx = cache.text_to_index.get(text)
            if idx is None:
                rows.append(_empty_vector(cache))
            else:
                sbert_vec = cache.sbert[int(idx)]
                fin_idx = cache.finbert_text_to_index.get(text)
                if fin_idx is None:
                    fin_vec = np.zeros(cache.finbert.shape[1], dtype=np.float32)
                else:
                    fin_vec = cache.finbert[int(fin_idx)]
                rows.append(_concat_embeddings(fin_vec, sbert_vec))
            continue

        headline_idx = _headline_indices(row["headlines"], cache.text_to_index)
        scores = score_headline_indices(
            headline_indices=headline_idx,
            cache=cache,
            relevance_weight=relevance_weight,
            finbert_top_n=finbert_top_n,
        )

        if variant == "f1_top1":
            rows.append(_top_k_weighted_pool(headline_idx, scores, top_k=1, cache=cache))
        elif variant == "f2_top3":
            rows.append(_top_k_weighted_pool(headline_idx, scores, top_k=3, cache=cache))
        elif variant == "f3_top5":
            rows.append(_top_k_weighted_pool(headline_idx, scores, top_k=5, cache=cache))
        elif variant == "f4_all_weighted":
            rows.append(weighted_pool(headline_idx, scores, cache))
        else:
            raise ValueError(f"Unsupported variant: {variant}")

    return np.vstack(rows)
