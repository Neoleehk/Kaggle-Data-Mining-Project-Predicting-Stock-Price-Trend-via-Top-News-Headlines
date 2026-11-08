from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .relevance import DEFAULT_ANCHORS


TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9'_-]*")
FINANCE_KEYWORDS = {
    "stock",
    "stocks",
    "market",
    "markets",
    "djia",
    "dow",
    "index",
    "indices",
    "equity",
    "equities",
    "shares",
    "investor",
    "investors",
    "trading",
    "trade",
    "earnings",
    "guidance",
    "profit",
    "loss",
    "revenue",
    "forecast",
    "outlook",
    "economy",
    "economic",
    "inflation",
    "deflation",
    "interest",
    "rates",
    "federal",
    "fed",
    "central",
    "bank",
    "policy",
    "tariff",
    "sanctions",
    "oil",
    "energy",
    "volatility",
    "risk",
    "recession",
    "gdp",
    "employment",
    "jobs",
}


@dataclass(frozen=True)
class LightweightRelevanceFilter:
    vectorizer: TfidfVectorizer
    anchor_matrix: object
    keyword_weight: float
    tfidf_weight: float

    def score(self, headlines: list[str]) -> np.ndarray:
        if not headlines:
            return np.empty(0, dtype=np.float32)

        matrix = self.vectorizer.transform(headlines)
        sim_matrix = (matrix @ self.anchor_matrix.T).toarray()
        tfidf_scores = np.asarray(sim_matrix.max(axis=1), dtype=np.float32)
        keyword_scores = np.array([_keyword_score(h) for h in headlines], dtype=np.float32)
        combined = self.keyword_weight * keyword_scores + self.tfidf_weight * tfidf_scores
        return combined.astype(np.float32)

    def select_top(self, headlines: list[str], top_k: int) -> list[str]:
        if not headlines:
            return []
        scores = self.score(headlines)
        order = np.argsort(scores)[::-1]
        keep = [headlines[int(i)] for i in order[: min(top_k, len(order))]]
        return keep


def _keyword_score(text: str) -> float:
    tokens = TOKEN_PATTERN.findall(text.lower())
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in FINANCE_KEYWORDS)
    return hits / np.sqrt(len(tokens) + 1.0)


def fit_lightweight_filter(
    train_headlines: list[str],
    keyword_weight: float = 0.4,
    tfidf_weight: float = 0.6,
    min_df: int = 2,
    max_features: int = 30000,
    anchors: list[str] | None = None,
) -> LightweightRelevanceFilter:
    if not 0 <= keyword_weight <= 1:
        raise ValueError("keyword_weight must be in [0, 1].")
    if not 0 <= tfidf_weight <= 1:
        raise ValueError("tfidf_weight must be in [0, 1].")

    use_anchors = anchors if anchors is not None else DEFAULT_ANCHORS
    corpus = [h for h in train_headlines if h] + use_anchors
    if not corpus:
        corpus = use_anchors

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=min_df,
        max_features=max_features,
    )
    try:
        vectorizer.fit(corpus)
    except ValueError:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_features=max_features,
        )
        vectorizer.fit(corpus)

    anchor_matrix = vectorizer.transform(use_anchors)
    return LightweightRelevanceFilter(
        vectorizer=vectorizer,
        anchor_matrix=anchor_matrix,
        keyword_weight=keyword_weight,
        tfidf_weight=tfidf_weight,
    )
