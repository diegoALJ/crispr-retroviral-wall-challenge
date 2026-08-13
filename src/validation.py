from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import pandas as pd

from src.config import CFG


def iter_lofo_splits(
    df: pd.DataFrame,
    cfg: CFG,
) -> Iterator[Tuple[str, np.ndarray, np.ndarray]]:
    if cfg.family_col not in df.columns:
        raise ValueError(
            f"Family column '{cfg.family_col}' not found."
        )

    families = sorted(
        df[cfg.family_col]
        .dropna()
        .unique()
        .tolist()
    )

    for family in families:
        val_mask = (
            df[cfg.family_col].values == family
        )

        train_idx = np.where(~val_mask)[0]
        val_idx = np.where(val_mask)[0]

        yield family, train_idx, val_idx


def describe_lofo_splits(
    df: pd.DataFrame,
    cfg: CFG,
) -> pd.DataFrame:
    rows = []

    for family, train_idx, val_idx in iter_lofo_splits(
        df,
        cfg,
    ):
        rows.append(
            {
                "held_out_family": family,
                "n_train": len(train_idx),
                "n_validation": len(val_idx),
            }
        )

    return pd.DataFrame(rows)
