from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import EsmModel, EsmTokenizer

from src.config import CFG
from src.metrics import blend_rank_scores


def get_device() -> str:
    return (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def load_esm2_backbone(cfg: CFG):
    model_source = (
        cfg.esm_local_dir
        if cfg.esm_local_dir is not None
        else cfg.esm_model_name
    )

    device = get_device()

    print(f"Loading ESM-2 from: {model_source}")
    print(f"Using device: {device}")

    tokenizer = EsmTokenizer.from_pretrained(
        model_source
    )

    model = EsmModel.from_pretrained(
        model_source
    )

    model.to(device)
    model.eval()

    if (
        cfg.esm_use_fp16
        and device == "cuda"
    ):
        model.half()

    return tokenizer, model, device


def mean_pool_esm_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    pool_mask = attention_mask.bool().clone()

    # Exclude BOS / CLS token.
    pool_mask[:, 0] = False

    # Exclude EOS token.
    valid_lengths = attention_mask.sum(dim=1)

    for i in range(pool_mask.size(0)):
        eos_idx = int(
            valid_lengths[i].item()
        ) - 1

        if eos_idx >= 0:
            pool_mask[i, eos_idx] = False

    pool_mask = pool_mask.unsqueeze(-1)

    masked_hidden = (
        hidden_states * pool_mask
    )

    denominator = (
        pool_mask
        .sum(dim=1)
        .clamp(min=1)
    )

    return (
        masked_hidden.sum(dim=1)
        / denominator
    )


def extract_sequence_embeddings(
    df: pd.DataFrame,
    cfg: CFG,
    tokenizer,
    model,
    device: str,
    cache_path: Optional[str | Path] = None,
) -> np.ndarray:
    if cache_path is not None:
        cache_path = Path(cache_path)

        if cache_path.exists():
            print(
                f"Loading cached embeddings: "
                f"{cache_path}"
            )

            return np.load(cache_path)

    sequences = (
        df[cfg.seq_col]
        .astype(str)
        .tolist()
    )

    embeddings = []

    with torch.inference_mode():
        for start in range(
            0,
            len(sequences),
            cfg.esm_batch_size,
        ):
            batch_sequences = sequences[
                start:
                start + cfg.esm_batch_size
            ]

            encoded = tokenizer(
                batch_sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg.esm_max_length,
            )

            encoded = {
                key: value.to(device)
                for key, value
                in encoded.items()
            }

            outputs = model(**encoded)

            pooled = (
                mean_pool_esm_hidden_states(
                    hidden_states=(
                        outputs.last_hidden_state
                    ),
                    attention_mask=(
                        encoded["attention_mask"]
                    ),
                )
            )

            embeddings.append(
                pooled
                .detach()
                .float()
                .cpu()
                .numpy()
            )

    embedding_matrix = np.concatenate(
        embeddings,
        axis=0,
    ).astype(np.float32)

    if cache_path is not None:
        cache_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            cache_path,
            embedding_matrix,
        )

    return embedding_matrix


def build_embedding_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: CFG,
):
    tokenizer, model, device = (
        load_esm2_backbone(cfg)
    )

    cache_dir = Path(
        cfg.sequence_cache_dir
    )

    train_cache = (
        cache_dir
        / "train_sequence_embeddings.npy"
    )

    test_cache = (
        cache_dir
        / "test_sequence_embeddings.npy"
    )

    train_embeddings = (
        extract_sequence_embeddings(
            train_df,
            cfg,
            tokenizer,
            model,
            device,
            train_cache,
        )
    )

    test_embeddings = (
        extract_sequence_embeddings(
            test_df,
            cfg,
            tokenizer,
            model,
            device,
            test_cache,
        )
    )

    print(
        "Train embeddings:",
        train_embeddings.shape,
    )

    print(
        "Test embeddings:",
        test_embeddings.shape,
    )

    # Release GPU memory after feature extraction.
    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        train_embeddings,
        test_embeddings,
    )


def build_sequence_classifier(
    cfg: CFG,
) -> Pipeline:
    steps = [
        ("scaler", StandardScaler())
    ]

    if cfg.emb_use_pca:
        steps.append(
            (
                "pca",
                PCA(
                    n_components=(
                        cfg.emb_pca_n_components
                    ),
                    random_state=cfg.seed,
                ),
            )
        )

    steps.append(
        (
            "model",
            LogisticRegression(
                C=cfg.emb_logreg_C,
                max_iter=(
                    cfg.emb_logreg_max_iter
                ),
                class_weight=(
                    cfg.emb_logreg_class_weight
                ),
                random_state=cfg.seed,
            ),
        )
    )

    return Pipeline(steps)


def build_sequence_regressor(
    cfg: CFG,
) -> Pipeline:
    steps = [
        ("scaler", StandardScaler())
    ]

    if cfg.emb_use_pca:
        steps.append(
            (
                "pca",
                PCA(
                    n_components=(
                        cfg.emb_pca_n_components
                    ),
                    random_state=cfg.seed,
                ),
            )
        )

    steps.append(
        (
            "model",
            Ridge(
                alpha=cfg.emb_ridge_alpha
            ),
        )
    )

    return Pipeline(steps)


def fit_sequence_branch(
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: CFG,
):
    y_cls = train_df[
        cfg.target_class
    ].values

    y_reg = train_df[
        cfg.target_reg
    ].values

    # Classification head
    classifier = (
        build_sequence_classifier(cfg)
    )

    classifier.fit(
        train_embeddings,
        y_cls,
    )

    p_active = classifier.predict_proba(
        test_embeddings
    )[:, 1]

    # Regression head
    regressor = (
        build_sequence_regressor(cfg)
    )

    y_reg_fit = y_reg.copy()

    if cfg.emb_use_log1p_reg_target:
        y_reg_fit = np.log1p(
            y_reg_fit
        )

    regressor.fit(
        train_embeddings,
        y_reg_fit,
    )

    reg_pred = regressor.predict(
        test_embeddings
    )

    if cfg.emb_use_log1p_reg_target:
        reg_pred = np.expm1(
            reg_pred
        )

    reg_pred = np.clip(
        reg_pred,
        cfg.reg_clip_min,
        cfg.reg_clip_max,
    )

    predicted_score = blend_rank_scores(
        p_active=p_active,
        regression_pred=reg_pred,
        classification_weight=(
            cfg.cls_blend_weight
        ),
        regression_weight=(
            cfg.reg_blend_weight
        ),
    )

    prediction_df = test_df[
        [cfg.id_col]
    ].copy()

    prediction_df[
        "p_active_sequence"
    ] = p_active

    prediction_df[
        "reg_pred_sequence"
    ] = reg_pred

    prediction_df[
        "predicted_score_sequence"
    ] = predicted_score

    model_bundle = {
        "classifier": classifier,
        "regressor": regressor,
    }

    return prediction_df, model_bundle
