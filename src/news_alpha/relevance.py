from __future__ import annotations

import numpy as np

from .embeddings import EmbeddingCache


DEFAULT_ANCHORS = [
    "stock market movement",
    "equity index returns",
    "macroeconomic policy impact",
    "corporate earnings guidance",
    "market risk sentiment",
]


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores)


def _anchor_indices(cache: EmbeddingCache, anchors: list[str] | None = None) -> np.ndarray:
    use_anchors = anchors if anchors is not None else cache.anchors
    if not use_anchors:
        raise ValueError("No anchors provided and cache does not contain anchors.")

    missing = [anchor for anchor in use_anchors if anchor not in cache.text_to_index]
    if missing:
        raise ValueError(f"Anchors missing from cache: {missing}")

    return np.array([cache.text_to_index[a] for a in use_anchors], dtype=int)


def _finbert_anchor_indices(cache: EmbeddingCache, anchors: list[str]) -> np.ndarray:
    missing = [anchor for anchor in anchors if anchor not in cache.finbert_text_to_index]
    if missing:
        raise ValueError(f"Anchors missing from FinBERT cache: {missing}")
    return np.array([cache.finbert_text_to_index[a] for a in anchors], dtype=int)


def score_headline_indices(
    headline_indices: np.ndarray,
    cache: EmbeddingCache,
    relevance_weight: float = 0.5,
    anchors: list[str] | None = None,
    finbert_top_n: int | None = None,
) -> np.ndarray:
    if not 0 <= relevance_weight <= 1:
        raise ValueError("relevance_weight must be in [0, 1].")
    if headline_indices.size == 0:
        return np.empty(0, dtype=np.float32)

    anchor_idx = _anchor_indices(cache, anchors)
    anchor_texts = (anchors if anchors is not None else cache.anchors)
    fin_anchor = cache.finbert[_finbert_anchor_indices(cache, anchor_texts)]
    sbert_anchor = cache.sbert[anchor_idx]

    sbert_scores = cache.sbert[headline_indices] @ sbert_anchor.T
    sbert_max = np.max(sbert_scores, axis=1)

    if relevance_weight <= 0:
        return sbert_max.astype(np.float32)

    combined = sbert_max.astype(np.float32).copy()
    if finbert_top_n is None or finbert_top_n >= len(headline_indices):
        target_positions = np.arange(len(headline_indices), dtype=int)
    else:
        target_positions = np.argsort(sbert_max)[::-1][:finbert_top_n]

    for pos in target_positions:
        headline_idx = int(headline_indices[pos])
        text = cache.texts[headline_idx]
        fin_idx = cache.finbert_text_to_index.get(text)
        if fin_idx is None:
            continue

        fin_vector = cache.finbert[fin_idx]
        fin_score = float(np.max(fin_vector @ fin_anchor.T))
        combined[pos] = relevance_weight * fin_score + (1.0 - relevance_weight) * sbert_max[pos]

    return combined


def weighted_pool(
    indices: np.ndarray,
    scores: np.ndarray,
    cache: EmbeddingCache,
    combine_mode: str = "concat",
) -> np.ndarray:
    if indices.size == 0:
        fin_dim = cache.finbert.shape[1]
        sbert_dim = cache.sbert.shape[1]
        if combine_mode == "concat":
            return np.zeros(fin_dim + sbert_dim, dtype=np.float32)
        raise ValueError(f"Unsupported combine_mode: {combine_mode}")

    weights = _softmax(scores)
    sbert_vec = np.average(cache.sbert[indices], axis=0, weights=weights)
    fin_dim = cache.finbert.shape[1]
    fin_weighted = np.zeros(fin_dim, dtype=np.float32)
    fin_weight_total = 0.0
    for idx, weight in zip(indices, weights):
        text = cache.texts[int(idx)]
        fin_idx = cache.finbert_text_to_index.get(text)
        if fin_idx is None:
            continue
        fin_weighted += weight * cache.finbert[int(fin_idx)]
        fin_weight_total += float(weight)

    if fin_weight_total > 0:
        fin_vec = fin_weighted / fin_weight_total
    else:
        fin_vec = np.zeros(fin_dim, dtype=np.float32)

    if combine_mode == "concat":
        return np.concatenate([fin_vec, sbert_vec]).astype(np.float32)
    raise ValueError(f"Unsupported combine_mode: {combine_mode}")
