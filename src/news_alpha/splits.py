from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FoldRange:
    train_start: int
    train_end: int
    val_start: int
    val_end: int


@dataclass(frozen=True)
class TemporalSplitPlan:
    train_val_start: int
    train_val_end: int
    test_start: int
    test_end: int
    folds: Sequence[FoldRange]


DEFAULT_PLAN = TemporalSplitPlan(
    train_val_start=1,
    train_val_end=1591,
    test_start=1592,
    test_end=1989,
    folds=(
        FoldRange(train_start=1, train_end=835, val_start=836, val_end=1087),
        FoldRange(train_start=1, train_end=1087, val_start=1088, val_end=1339),
        FoldRange(train_start=1, train_end=1339, val_start=1340, val_end=1591),
    ),
)


def _row_slice_to_iloc(start: int, end: int) -> np.ndarray:
    if start < 1 or end < start:
        raise ValueError(f"Invalid row range: start={start}, end={end}")
    return np.arange(start - 1, end)


def validate_plan(df: pd.DataFrame, plan: TemporalSplitPlan) -> None:
    total_rows = len(df)
    if total_rows < plan.train_val_end:
        raise ValueError(
            f"Dataset has {total_rows} rows, but split plan expects at least {plan.train_val_end} rows."
        )

    dates = df["Date"]
    if not dates.is_monotonic_increasing:
        raise ValueError("DataFrame must be sorted by Date ascending before splitting.")

    for fold in plan.folds:
        if fold.val_end > total_rows:
            raise ValueError(
                f"Fold validation range ends at row {fold.val_end}, "
                f"but dataset has only {total_rows} rows."
            )


def build_indices(df: pd.DataFrame, plan: TemporalSplitPlan = DEFAULT_PLAN) -> dict[str, object]:
    validate_plan(df, plan)
    total_rows = len(df)
    effective_test_end = min(plan.test_end, total_rows)

    cv_folds = []
    for fold in plan.folds:
        cv_folds.append(
            {
                "train": _row_slice_to_iloc(fold.train_start, fold.train_end),
                "val": _row_slice_to_iloc(fold.val_start, fold.val_end),
                "train_range": (fold.train_start, fold.train_end),
                "val_range": (fold.val_start, fold.val_end),
            }
        )

    return {
        "train_val": _row_slice_to_iloc(plan.train_val_start, plan.train_val_end),
        "test": _row_slice_to_iloc(plan.test_start, effective_test_end),
        "train_val_range": (plan.train_val_start, plan.train_val_end),
        "test_range": (plan.test_start, effective_test_end),
        "cv_folds": cv_folds,
    }
