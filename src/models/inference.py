from __future__ import annotations

import argparse
from pathlib import Path

from src.config import CFG
from src.data import load_train_test
from src.features import (
    load_feature_sets,
    validate_feature_sets,
)
from src.models.ensemble import (
    blend_model_branches,
)
from src.models.sequence import (
    build_embedding_matrices,
    fit_sequence_branch,
)
from src.models.tabular import (
    fit_tabular_ensemble,
)
from src.utils import (
    ensure_dir,
    seed_everything,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate Retroviral Wall "
            "Challenge predictions."
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
    # Branch A: CatBoost tabular ensemble
    # ---------------------------------------------------------
    (
        tabular_pred_df,
        _,
        _,
    ) = fit_tabular_ensemble(
        train_df=train_df,
        test_df=test_df,
        feature_sets=feature_sets,
        cfg=cfg,
    )

    # ---------------------------------------------------------
    # Branch B: ESM-2 sequence model
    # ---------------------------------------------------------
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
        _,
    ) = fit_sequence_branch(
        train_embeddings=train_embeddings,
        test_embeddings=test_embeddings,
        train_df=train_df,
        test_df=test_df,
        cfg=cfg,
    )

    # ---------------------------------------------------------
    # Final multimodal ensemble
    # ---------------------------------------------------------
    final_df = blend_model_branches(
        tabular_pred_df=tabular_pred_df,
        sequence_pred_df=sequence_pred_df,
        cfg=cfg,
    )

    submission_df = final_df[
        [cfg.id_col, "predicted_score"]
    ].copy()

    submission_df.columns = [
        cfg.submission_id_col,
        cfg.submission_pred_col,
    ]

    output_dir = ensure_dir(
        cfg.output_dir
    )

    output_path = (
        output_dir
        / cfg.submission_filename
    )

    submission_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Submission saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
