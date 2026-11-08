from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_FINBERT_MODEL = "ProsusAI/finbert"
DEFAULT_SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class EmbeddingCache:
    texts: list[str]
    sbert: np.ndarray
    finbert_texts: list[str]
    finbert: np.ndarray
    text_to_index: dict[str, int]
    finbert_text_to_index: dict[str, int]
    finbert_model: str
    sbert_model: str
    anchors: list[str]
    strategy: str
    finbert_top_n: int


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def _chunk(values: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(values), size):
        yield values[i : i + size]


def _select_device(device: str | None) -> str:
    if device:
        return device

    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def encode_finbert(
    texts: list[str],
    model_name: str = DEFAULT_FINBERT_MODEL,
    batch_size: int = 64,
    max_length: int = 96,
    device: str | None = None,
) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    run_device = _select_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(run_device)
    model.eval()

    outputs: list[np.ndarray] = []
    for batch in _chunk(texts, batch_size):
        encoded = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(run_device) for k, v in encoded.items()}
        with torch.inference_mode():
            last_hidden = model(**encoded).last_hidden_state
            cls_vectors = last_hidden[:, 0, :].detach().cpu().numpy()
            outputs.append(cls_vectors.astype(np.float32))

    matrix = np.vstack(outputs)
    return _normalize_rows(matrix)


def encode_sbert(
    texts: list[str],
    model_name: str = DEFAULT_SBERT_MODEL,
    batch_size: int = 128,
    device: str | None = None,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    run_device = _select_device(device)
    model = SentenceTransformer(model_name, device=run_device)
    matrix = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return matrix.astype(np.float32)


def collect_unique_texts(df: pd.DataFrame, anchors: list[str]) -> list[str]:
    unique = set(anchors)
    for text in df["daily_text"].tolist():
        if text:
            unique.add(text)
    for headlines in df["headlines"].tolist():
        for headline in headlines:
            if headline:
                unique.add(headline)
    return sorted(unique)


def _select_finbert_texts_two_stage(
    df: pd.DataFrame,
    texts: list[str],
    text_to_index: dict[str, int],
    sbert: np.ndarray,
    anchors: list[str],
    finbert_top_n: int,
) -> list[str]:
    if finbert_top_n < 1:
        raise ValueError("finbert_top_n must be >= 1 for two_stage strategy.")

    missing_anchors = [a for a in anchors if a not in text_to_index]
    if missing_anchors:
        raise ValueError(f"Anchors missing from text list: {missing_anchors}")

    anchor_idx = np.array([text_to_index[a] for a in anchors], dtype=int)
    anchor_vectors = sbert[anchor_idx]

    keep: set[str] = set(anchors)
    for day_text in df["daily_text"].tolist():
        if day_text:
            keep.add(day_text)

    for headlines in df["headlines"].tolist():
        indices = np.array(
            [text_to_index[h] for h in headlines if h in text_to_index],
            dtype=int,
        )
        if indices.size == 0:
            continue

        sims = sbert[indices] @ anchor_vectors.T
        max_scores = np.max(sims, axis=1)
        top_positions = np.argsort(max_scores)[::-1][: min(finbert_top_n, len(indices))]
        top_indices = indices[top_positions]
        for idx in top_indices:
            keep.add(texts[int(idx)])

    return sorted(keep)


def build_embedding_cache(
    df: pd.DataFrame,
    texts: list[str],
    anchors: list[str],
    strategy: str = "full",
    finbert_top_n: int = 5,
    finbert_model: str = DEFAULT_FINBERT_MODEL,
    sbert_model: str = DEFAULT_SBERT_MODEL,
    finbert_batch_size: int = 64,
    sbert_batch_size: int = 128,
    max_length: int = 96,
    device: str | None = None,
) -> EmbeddingCache:
    text_to_index = {text: idx for idx, text in enumerate(texts)}
    sbert = encode_sbert(
        texts=texts,
        model_name=sbert_model,
        batch_size=sbert_batch_size,
        device=device,
    )

    if strategy == "full":
        finbert_texts = texts
    elif strategy == "two_stage":
        finbert_texts = _select_finbert_texts_two_stage(
            df=df,
            texts=texts,
            text_to_index=text_to_index,
            sbert=sbert,
            anchors=anchors,
            finbert_top_n=finbert_top_n,
        )
    else:
        raise ValueError("strategy must be one of {'full', 'two_stage'}")

    finbert = encode_finbert(
        texts=finbert_texts,
        model_name=finbert_model,
        batch_size=finbert_batch_size,
        max_length=max_length,
        device=device,
    )
    finbert_text_to_index = {text: idx for idx, text in enumerate(finbert_texts)}

    return EmbeddingCache(
        texts=texts,
        sbert=sbert,
        finbert_texts=finbert_texts,
        finbert=finbert,
        text_to_index=text_to_index,
        finbert_text_to_index=finbert_text_to_index,
        finbert_model=finbert_model,
        sbert_model=sbert_model,
        anchors=anchors,
        strategy=strategy,
        finbert_top_n=finbert_top_n,
    )


def save_embedding_cache(cache: EmbeddingCache, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "sbert.npy", cache.sbert)
    np.save(output_dir / "finbert.npy", cache.finbert)

    with (output_dir / "texts.jsonl").open("w", encoding="utf-8") as f:
        for text in cache.texts:
            f.write(json.dumps(text, ensure_ascii=False) + "\n")
    with (output_dir / "finbert_texts.jsonl").open("w", encoding="utf-8") as f:
        for text in cache.finbert_texts:
            f.write(json.dumps(text, ensure_ascii=False) + "\n")

    metadata = {
        "finbert_model": cache.finbert_model,
        "sbert_model": cache.sbert_model,
        "rows_sbert": len(cache.texts),
        "rows_finbert": len(cache.finbert_texts),
        "finbert_dim": int(cache.finbert.shape[1]),
        "sbert_dim": int(cache.sbert.shape[1]),
        "anchors": cache.anchors,
        "strategy": cache.strategy,
        "finbert_top_n": cache.finbert_top_n,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def load_embedding_cache(cache_dir: Path) -> EmbeddingCache:
    sbert = np.load(cache_dir / "sbert.npy")
    finbert = np.load(cache_dir / "finbert.npy")
    texts = [
        json.loads(line)
        for line in (cache_dir / "texts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))

    finbert_texts_path = cache_dir / "finbert_texts.jsonl"
    if finbert_texts_path.exists():
        finbert_texts = [
            json.loads(line)
            for line in finbert_texts_path.read_text(encoding="utf-8").splitlines()
        ]
    else:
        finbert_texts = texts

    if sbert.shape[0] != len(texts):
        raise ValueError("Cache mismatch: SBERT row count does not match texts.")
    if finbert.shape[0] != len(finbert_texts):
        raise ValueError("Cache mismatch: FinBERT row count does not match finbert_texts.")

    text_to_index = {text: idx for idx, text in enumerate(texts)}
    finbert_text_to_index = {text: idx for idx, text in enumerate(finbert_texts)}
    return EmbeddingCache(
        texts=texts,
        sbert=sbert,
        finbert_texts=finbert_texts,
        finbert=finbert,
        text_to_index=text_to_index,
        finbert_text_to_index=finbert_text_to_index,
        finbert_model=metadata["finbert_model"],
        sbert_model=metadata["sbert_model"],
        anchors=metadata.get("anchors", []),
        strategy=metadata.get("strategy", "full"),
        finbert_top_n=int(metadata.get("finbert_top_n", 0)),
    )
