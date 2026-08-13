from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from src.config import CFG


def load_train_test(cfg: CFG) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(cfg.train_path)
    test_df = pd.read_csv(cfg.test_path)

    validate_core_columns(train_df, test_df, cfg)

    return train_df, test_df


def load_family_splits(cfg: CFG) -> pd.DataFrame:
    path = Path(cfg.family_splits_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Family splits file not found: {path}"
        )

    return pd.read_csv(path)


def load_feature_dictionary(cfg: CFG) -> pd.DataFrame:
    path = Path(cfg.feature_dictionary_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Feature dictionary not found: {path}"
        )

    return pd.read_csv(path)


def validate_core_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: CFG,
) -> None:
    required_train = {
        cfg.id_col,
        cfg.seq_col,
        cfg.family_col,
        cfg.target_class,
        cfg.target_reg,
    }

    required_test = {
        cfg.id_col,
        cfg.seq_col,
    }

    missing_train = required_train - set(train_df.columns)
    missing_test = required_test - set(test_df.columns)

    if missing_train:
        raise ValueError(
            f"Missing required train columns: {sorted(missing_train)}"
        )

    if missing_test:
        raise ValueError(
            f"Missing required test columns: {sorted(missing_test)}"
        )

    if train_df[cfg.id_col].duplicated().any():
        raise ValueError(
            f"Duplicate IDs detected in train column '{cfg.id_col}'."
        )

    if test_df[cfg.id_col].duplicated().any():
        raise ValueError(
            f"Duplicate IDs detected in test column '{cfg.id_col}'."
        )


def get_numeric_feature_columns(
    train_df: pd.DataFrame,
    cfg: CFG,
) -> list[str]:
    excluded = {
        cfg.id_col,
        cfg.seq_col,
        cfg.family_col,
        cfg.target_class,
        cfg.target_reg,
    }

    return [
        col
        for col in train_df.select_dtypes(include="number").columns
        if col not in excluded
    ]
