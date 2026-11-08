from __future__ import annotations

from dataclasses import dataclass
from html import unescape
from pathlib import Path
import re
from typing import Iterable

import pandas as pd

TOP_COLUMNS = [f"Top{i}" for i in range(1, 26)]
BYTE_PREFIX_PATTERN = re.compile(r"""^\s*b(['"])(.*)\1\s*$""")


@dataclass(frozen=True)
class DataPaths:
    raw_csv: Path
    clean_csv: Path


def _clean_piece(text: str) -> str:
    value = "" if text is None else str(text).strip()
    if not value:
        return ""

    match = BYTE_PREFIX_PATTERN.match(value)
    if match:
        value = match.group(2)

    value = unescape(value)
    value = value.replace("\\'", "'").replace('\\"', '"')
    value = " ".join(value.split())
    return value.strip()


def _clean_non_empty_texts(texts: Iterable[str]) -> list[str]:
    cleaned = []
    for text in texts:
        normalized = _clean_piece(text)
        if normalized:
            cleaned.append(normalized)
    return cleaned


def load_daily_news(raw_csv_path: Path, clean_csv_path: Path) -> pd.DataFrame:
    raw_df = pd.read_csv(raw_csv_path)
    clean_df = pd.read_csv(clean_csv_path)

    raw_df["Date"] = pd.to_datetime(raw_df["Date"], format="%Y-%m-%d")
    clean_df["Date"] = pd.to_datetime(clean_df["Date"], format="%Y-%m-%d")
    raw_df["Label"] = raw_df["Label"].astype(int)
    clean_df["Label"] = clean_df["Label"].astype(int)

    missing_columns = [col for col in TOP_COLUMNS if col not in raw_df.columns]
    if missing_columns:
        raise ValueError(f"Missing headline columns in raw CSV: {missing_columns}")

    for col in TOP_COLUMNS:
        raw_df[col] = raw_df[col].fillna("")

    raw_df["headlines"] = raw_df[TOP_COLUMNS].apply(_clean_non_empty_texts, axis=1)
    clean_df["daily_text"] = clean_df["text"].fillna("").map(_clean_piece)

    merged = (
        raw_df[["Date", "Label", "headlines"]]
        .merge(clean_df[["Date", "Label", "daily_text"]], on=["Date", "Label"], how="inner")
        .sort_values("Date")
        .reset_index(drop=True)
    )

    merged["headline_count"] = merged["headlines"].map(len)
    merged.insert(0, "row_id", merged.index + 1)
    return merged


def with_target(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode not in {"same_day", "next_day"}:
        raise ValueError("mode must be one of {'same_day', 'next_day'}")

    data = df.copy()
    if mode == "same_day":
        data["Target"] = data["Label"]
        return data

    data["Target"] = data["Label"].shift(-1)
    data = data.iloc[:-1].copy()
    data["Target"] = data["Target"].astype(int)
    return data

