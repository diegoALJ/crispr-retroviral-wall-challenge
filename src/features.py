from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_feature_sets(path: str | Path) -> Dict[str, List[str]]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Feature set configuration not found: {path}"
        )

    with path.open("r", encoding="utf-8") as f:
        feature_sets = json.load(f)

    if not isinstance(feature_sets, dict):
        raise ValueError(
            "Feature set file must contain a dictionary."
        )

    return feature_sets


def resolve_feature_set(
    feature_set_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_sets: Dict[str, List[str]],
) -> List[str]:
    if feature_set_name not in feature_sets:
        raise KeyError(
            f"Unknown feature set: {feature_set_name}"
        )

    requested = feature_sets[feature_set_name]

    available = [
        feature
        for feature in requested
        if feature in train_df.columns
        and feature in test_df.columns
        and pd.api.types.is_numeric_dtype(train_df[feature])
    ]

    missing = sorted(set(requested) - set(available))

    if missing:
        print(
            f"[{feature_set_name}] "
            f"{len(missing)} features unavailable and ignored."
        )

    if not available:
        raise ValueError(
            f"No usable features found for '{feature_set_name}'."
        )

    return available


def validate_feature_sets(
    feature_sets: Dict[str, List[str]],
    required_sets: List[str],
) -> None:
    missing = [
        name
        for name in required_sets
        if name not in feature_sets
    ]

    if missing:
        raise ValueError(
            f"Required feature sets not found: {missing}"
        )
