from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from catboost import (
    CatBoostClassifier,
    CatBoostRegressor,
)

from src.config import CFG
from src.features import resolve_feature_set
from src.metrics import blend_rank_scores


def build_tabular_classifier(
    cfg: CFG,
) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=cfg.cb_cls_iterations,
        depth=cfg.cb_cls_depth,
        learning_rate=cfg.cb_cls_learning_rate,
        l2_leaf_reg=cfg.cb_cls_l2_leaf_reg,
        loss_function=cfg.cb_cls_loss_function,
        eval_metric=cfg.cb_cls_eval_metric,
        random_seed=cfg.seed,
        verbose=False,
    )


def build_tabular_regressor(
    cfg: CFG,
) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=cfg.cb_reg_iterations,
        depth=cfg.cb_reg_depth,
        learning_rate=cfg.cb_reg_learning_rate,
        l2_leaf_reg=cfg.cb_reg_l2_leaf_reg,
        loss_function=cfg.cb_reg_loss_function,
        random_seed=cfg.seed,
        verbose=False,
    )


def clip_regression_predictions(
    predictions: np.ndarray,
    cfg: CFG,
) -> np.ndarray:
    return np.clip(
        predictions,
        cfg.reg_clip_min,
        cfg.reg_clip_max,
    )


def fit_single_feature_set(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_set_name: str,
    feature_sets: Dict[str, List[str]],
    cfg: CFG,
):
    features = resolve_feature_set(
        feature_set_name,
        train_df,
        test_df,
        feature_sets,
    )

    X_train = train_df[features]
    X_test = test_df[features]

    y_cls = train_df[cfg.target_class].values
    y_reg = train_df[cfg.target_reg].values

    # ---------------------------------------------------------
    # Classification
    # ---------------------------------------------------------
    classifier = build_tabular_classifier(cfg)

    classifier.fit(
        X_train,
        y_cls,
    )

    p_active = classifier.predict_proba(
        X_test
    )[:, 1]

    # ---------------------------------------------------------
    # Regression
    # ---------------------------------------------------------
    regressor = build_tabular_regressor(cfg)

    if cfg.reg_train_mode == "positives_only":
        reg_mask = y_cls == 1
    elif cfg.reg_train_mode == "all_rows":
        reg_mask = np.ones(
            len(train_df),
            dtype=bool,
        )
    else:
        raise ValueError(
            f"Unknown reg_train_mode: {cfg.reg_train_mode}"
        )

    X_reg = X_train.loc[reg_mask]
    y_reg_fit = y_reg[reg_mask]

    if cfg.use_log1p_reg_target:
        y_reg_fit = np.log1p(y_reg_fit)

    regressor.fit(
        X_reg,
        y_reg_fit,
    )

    reg_pred = regressor.predict(X_test)

    if cfg.use_log1p_reg_target:
        reg_pred = np.expm1(reg_pred)

    reg_pred = clip_regression_predictions(
        reg_pred,
        cfg,
    )

    predicted_score = blend_rank_scores(
        p_active=p_active,
        regression_pred=reg_pred,
        classification_weight=cfg.cls_blend_weight,
        regression_weight=cfg.reg_blend_weight,
    )

    prediction_df = test_df[
        [cfg.id_col]
    ].copy()

    prediction_df["feature_set"] = (
        feature_set_name
    )
    prediction_df["p_active"] = p_active
    prediction_df["reg_pred"] = reg_pred
    prediction_df["predicted_score"] = (
        predicted_score
    )

    model_bundle = {
        "feature_set": feature_set_name,
        "features": features,
        "classifier": classifier,
        "regressor": regressor,
    }

    return prediction_df, model_bundle


def fit_tabular_ensemble(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_sets: Dict[str, List[str]],
    cfg: CFG,
):
    prediction_frames = []
    model_bundles = {}

    for feature_set_name in cfg.ensemble_feature_sets:
        print(
            f"Training tabular feature set: "
            f"{feature_set_name}"
        )

        pred_df, bundle = fit_single_feature_set(
            train_df=train_df,
            test_df=test_df,
            feature_set_name=feature_set_name,
            feature_sets=feature_sets,
            cfg=cfg,
        )

        prediction_frames.append(pred_df)
        model_bundles[feature_set_name] = bundle

    ensemble_df = test_df[
        [cfg.id_col]
    ].copy()

    score_columns = []

    for pred_df in prediction_frames:
        feature_set_name = (
            pred_df["feature_set"].iloc[0]
        )

        score_col = (
            f"score_{feature_set_name}"
        )

        score_columns.append(score_col)

        temp = pred_df[
            [cfg.id_col, "predicted_score"]
        ].rename(
            columns={
                "predicted_score": score_col
            }
        )

        ensemble_df = ensemble_df.merge(
            temp,
            on=cfg.id_col,
            how="left",
        )

    ensemble_df["predicted_score"] = (
        ensemble_df[score_columns]
        .mean(axis=1)
    )

    return (
        ensemble_df,
        model_bundles,
        prediction_frames,
    )
