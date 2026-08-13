from __future__ import annotations

import argparse
from pathlib import Path

from src.config import CFG
from src.data import load_train_test
from src.features import (
    load_feature_sets,
    validate_feature_sets,
)
from src.models.sequence import (
    build_embedding_matrices,
    fit_sequence_branch,
)
from src.models.tabular import (
    fit_tabular_ensemble,
)
from src.utils import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train Retroviral Wall Challenge models."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
    )

    parser.add_argument(
        "--feature-sets",
        type=str,
        default="configs/feature_sets.json",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = CFG.from_yaml(
        args.config
    )

    seed_everything(cfg.seed)

    train_df, test_df = (
        load_train_test(cfg)
    )

    feature_sets = load_feature_sets(
        args.feature_sets
    )

    validate_feature_sets(
        feature_sets,
        cfg.ensemble_feature_sets,
    )

    # ---------------------------------------------------------
    # Branch A
    # ---------------------------------------------------------
    (
        tabular_pred_df,
        tabular_models,
        _,
    ) = fit_tabular_ensemble(
        train_df=train_df,
        test_df=test_df,
        feature_sets=feature_sets,
        cfg=cfg,
    )

    # ---------------------------------------------------------
    # Branch B
    # ---------------------------------------------------------
    if cfg.sequence_branch_enabled:
        (
            train_embeddings,
            test_embeddings,
        ) = build_embedding_matrices(
            train_df,
            test_df,
            cfg,
        )

        (
            sequence_pred_df,
            sequence_models,
        ) = fit_sequence_branch(
            train_embeddings,
            test_embeddings,
            train_df,
            test_df,
            cfg,
        )

        print(
            "Both model branches "
            "trained successfully."
        )

    else:
        print(
            "Sequence branch disabled. "
            "Tabular model trained only."
        )


if __name__ == "__main__":
    main()
