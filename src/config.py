from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class CFG:
    # ---------------------------------------------------------
    # Project
    # ---------------------------------------------------------
    project_name: str = "retroviral-wall-challenge"
    seed: int = 42
    output_dir: str = "outputs"

    # ---------------------------------------------------------
    # Data
    # ---------------------------------------------------------
    train_path: str = "data/train.csv"
    test_path: str = "data/test.csv"
    family_splits_path: str = "data/family_splits.csv"
    feature_dictionary_path: str = "data/feature_dictionary.csv"

    id_col: str = "rt_name"
    seq_col: str = "sequence"
    family_col: str = "rt_family"

    target_class: str = "active"
    target_reg: str = "pe_efficiency_pct"

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------
    validation_strategy: str = "LOFO"

    # ---------------------------------------------------------
    # Branch A: CatBoost
    # ---------------------------------------------------------
    ensemble_feature_sets: List[str] = field(
        default_factory=lambda: [
            "foldseek",
            "foldseek_plus_catalytic",
            "foldseek_plus_mechanistic",
        ]
    )

    cb_cls_iterations: int = 500
    cb_cls_depth: int = 4
    cb_cls_learning_rate: float = 0.03
    cb_cls_l2_leaf_reg: float = 5.0
    cb_cls_loss_function: str = "Logloss"
    cb_cls_eval_metric: str = "AUC"

    cb_reg_iterations: int = 500
    cb_reg_depth: int = 4
    cb_reg_learning_rate: float = 0.03
    cb_reg_l2_leaf_reg: float = 5.0
    cb_reg_loss_function: str = "RMSE"

    reg_train_mode: str = "all_rows"
    use_log1p_reg_target: bool = True

    reg_clip_min: float = 0.0
    reg_clip_max: float = 41.0

    cls_blend_weight: float = 0.35
    reg_blend_weight: float = 0.65

    # ---------------------------------------------------------
    # Branch B: ESM-2
    # ---------------------------------------------------------
    sequence_branch_enabled: bool = True

    esm_model_name: str = "facebook/esm2_t33_650M_UR50D"
    esm_local_dir: str | None = None

    esm_batch_size: int = 1
    esm_max_length: int = 1024
    esm_use_fp16: bool = False

    sequence_cache_dir: str = "cache/sequence_embeddings"

    emb_use_pca: bool = True
    emb_pca_n_components: int = 32

    emb_logreg_C: float = 0.5
    emb_logreg_max_iter: int = 3000
    emb_logreg_class_weight: str = "balanced"

    emb_ridge_alpha: float = 3.0
    emb_use_log1p_reg_target: bool = True

    # ---------------------------------------------------------
    # Final multimodal ensemble
    # ---------------------------------------------------------
    final_tabular_weight: float = 0.60
    final_sequence_weight: float = 0.40

    # ---------------------------------------------------------
    # Submission
    # ---------------------------------------------------------
    submission_id_col: str = "rt_name"
    submission_pred_col: str = "predicted_score"
    submission_filename: str = "submission.csv"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CFG":
        path = Path(path)

        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        cfg = cls()

        project = raw.get("project", {})
        cfg.project_name = project.get("name", cfg.project_name)
        cfg.seed = project.get("seed", cfg.seed)
        cfg.output_dir = project.get("output_dir", cfg.output_dir)

        data = raw.get("data", {})
        cfg.train_path = data.get("train_path", cfg.train_path)
        cfg.test_path = data.get("test_path", cfg.test_path)
        cfg.family_splits_path = data.get(
            "family_splits_path", cfg.family_splits_path
        )
        cfg.feature_dictionary_path = data.get(
            "feature_dictionary_path", cfg.feature_dictionary_path
        )

        cfg.id_col = data.get("id_column", cfg.id_col)
        cfg.seq_col = data.get("sequence_column", cfg.seq_col)
        cfg.family_col = data.get("family_column", cfg.family_col)

        cfg.target_class = data.get(
            "classification_target", cfg.target_class
        )
        cfg.target_reg = data.get(
            "regression_target", cfg.target_reg
        )

        validation = raw.get("validation", {})
        cfg.validation_strategy = validation.get(
            "strategy", cfg.validation_strategy
        )

        tabular = raw.get("tabular", {})

        cfg.ensemble_feature_sets = tabular.get(
            "feature_sets", cfg.ensemble_feature_sets
        )

        cls_cfg = tabular.get("classifier", {})
        cfg.cb_cls_iterations = cls_cfg.get(
            "iterations", cfg.cb_cls_iterations
        )
        cfg.cb_cls_depth = cls_cfg.get(
            "depth", cfg.cb_cls_depth
        )
        cfg.cb_cls_learning_rate = cls_cfg.get(
            "learning_rate", cfg.cb_cls_learning_rate
        )
        cfg.cb_cls_l2_leaf_reg = cls_cfg.get(
            "l2_leaf_reg", cfg.cb_cls_l2_leaf_reg
        )
        cfg.cb_cls_loss_function = cls_cfg.get(
            "loss_function", cfg.cb_cls_loss_function
        )
        cfg.cb_cls_eval_metric = cls_cfg.get(
            "eval_metric", cfg.cb_cls_eval_metric
        )

        reg_cfg = tabular.get("regressor", {})
        cfg.cb_reg_iterations = reg_cfg.get(
            "iterations", cfg.cb_reg_iterations
        )
        cfg.cb_reg_depth = reg_cfg.get(
            "depth", cfg.cb_reg_depth
        )
        cfg.cb_reg_learning_rate = reg_cfg.get(
            "learning_rate", cfg.cb_reg_learning_rate
        )
        cfg.cb_reg_l2_leaf_reg = reg_cfg.get(
            "l2_leaf_reg", cfg.cb_reg_l2_leaf_reg
        )
        cfg.cb_reg_loss_function = reg_cfg.get(
            "loss_function", cfg.cb_reg_loss_function
        )

        regression = tabular.get("regression", {})
        cfg.reg_train_mode = regression.get(
            "train_mode", cfg.reg_train_mode
        )
        cfg.use_log1p_reg_target = regression.get(
            "log1p_target", cfg.use_log1p_reg_target
        )
        cfg.reg_clip_min = regression.get(
            "clip_min", cfg.reg_clip_min
        )
        cfg.reg_clip_max = regression.get(
            "clip_max", cfg.reg_clip_max
        )

        score_blend = tabular.get("score_blend", {})
        cfg.cls_blend_weight = score_blend.get(
            "classification_weight", cfg.cls_blend_weight
        )
        cfg.reg_blend_weight = score_blend.get(
            "regression_weight", cfg.reg_blend_weight
        )

        sequence = raw.get("sequence", {})

        backbone = sequence.get("backbone", {})
        cfg.esm_model_name = backbone.get(
            "model_name", cfg.esm_model_name
        )
        cfg.esm_batch_size = backbone.get(
            "batch_size", cfg.esm_batch_size
        )
        cfg.esm_max_length = backbone.get(
            "max_length", cfg.esm_max_length
        )
        cfg.esm_use_fp16 = backbone.get(
            "use_fp16", cfg.esm_use_fp16
        )

        embedding = sequence.get("embedding", {})
        cfg.sequence_cache_dir = embedding.get(
            "cache_dir", cfg.sequence_cache_dir
        )

        dim_red = sequence.get("dimensionality_reduction", {})
        cfg.emb_use_pca = (
            dim_red.get("method", "pca").lower() == "pca"
        )
        cfg.emb_pca_n_components = dim_red.get(
            "n_components", cfg.emb_pca_n_components
        )

        seq_cls = sequence.get("classifier", {})
        cfg.emb_logreg_C = seq_cls.get("C", cfg.emb_logreg_C)
        cfg.emb_logreg_max_iter = seq_cls.get(
            "max_iter", cfg.emb_logreg_max_iter
        )
        cfg.emb_logreg_class_weight = seq_cls.get(
            "class_weight", cfg.emb_logreg_class_weight
        )

        seq_reg = sequence.get("regressor", {})
        cfg.emb_ridge_alpha = seq_reg.get(
            "alpha", cfg.emb_ridge_alpha
        )

        seq_regression = sequence.get("regression", {})
        cfg.emb_use_log1p_reg_target = seq_regression.get(
            "log1p_target", cfg.emb_use_log1p_reg_target
        )

        ensemble = raw.get("ensemble", {})
        cfg.final_tabular_weight = ensemble.get(
            "tabular_weight", cfg.final_tabular_weight
        )
        cfg.final_sequence_weight = ensemble.get(
            "sequence_weight", cfg.final_sequence_weight
        )

        submission = raw.get("submission", {})
        cfg.submission_id_col = submission.get(
            "id_column", cfg.submission_id_col
        )
        cfg.submission_pred_col = submission.get(
            "prediction_column", cfg.submission_pred_col
        )
        cfg.submission_filename = submission.get(
            "filename", cfg.submission_filename
        )

        return cfg
