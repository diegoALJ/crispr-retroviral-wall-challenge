from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CFG


def validate_ensemble_weights(
    cfg: CFG,
) -> None:
    total = (
        cfg.final_tabular_weight
        + cfg.final_sequence_weight
    )

    if not np.isclose(total, 1.0):
        raise ValueError(
            "Final branch weights must sum to 1."
        )


def blend_model_branches(
    tabular_pred_df: pd.DataFrame,
    sequence_pred_df: pd.DataFrame,
    cfg: CFG,
) -> pd.DataFrame:
    validate_ensemble_weights(cfg)

    tabular = tabular_pred_df[
        [cfg.id_col, "predicted_score"]
    ].rename(
        columns={
            "predicted_score":
            "predicted_score_tabular"
        }
    )

    sequence = sequence_pred_df[
        [
            cfg.id_col,
            "predicted_score_sequence",
        ]
    ]

    final_df = tabular.merge(
        sequence,
        on=cfg.id_col,
        how="inner",
        validate="one_to_one",
    )

    final_df["predicted_score"] = (
        cfg.final_tabular_weight
        * final_df["predicted_score_tabular"]
        +
        cfg.final_sequence_weight
        * final_df["predicted_score_sequence"]
    )

    return final_df
