"""
Shared utilities for the idSCD pipeline:
  * seeding / determinism
  * CSV loading + column normalisation, label encoding
  * BEE helpers: normalize_embeddings, embed_text_list, bee_scores, pearson
  * WeightedTrainer (class-weighted cross-entropy for the classification head)
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# Must be set before torch initialises CUDA for use_deterministic_algorithms()
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .config import GroupConfig


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:  # pragma: no cover
            print(f"[warn] could not enable deterministic algorithms: {e}")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_group_csv(group: GroupConfig, path: Path, need_label: bool = True) -> pd.DataFrame:
    """
    Load a training-configuration CSV and normalise column names so that the
    text lives in group.text_col and the label in group.label_col
    (e.g. a 'label_str' column is renamed to 'label').
    """
    df = pd.read_csv(path)

    for alias in group.text_col_aliases:
        if alias in df.columns and group.text_col not in df.columns:
            df = df.rename(columns={alias: group.text_col})
    for alias in group.label_col_aliases:
        if alias in df.columns and group.label_col not in df.columns:
            df = df.rename(columns={alias: group.label_col})

    if group.text_col not in df.columns:
        raise KeyError(f"{path}: text column '{group.text_col}' not found "
                       f"(columns: {list(df.columns)})")
    subset = [group.text_col]
    if need_label:
        if group.label_col not in df.columns:
            raise KeyError(f"{path}: label column '{group.label_col}' not found "
                           f"(columns: {list(df.columns)})")
        subset.append(group.label_col)

    df = df.dropna(subset=subset).copy()
    df[group.text_col] = df[group.text_col].astype(str)
    if need_label:
        df[group.label_col] = df[group.label_col].astype(str)
    return df


def encode_labels(df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, Dict[int, str], Dict[str, int]]:
    """LabelEncoder-equivalent: alphabetical label ids."""
    classes = sorted(df[label_col].unique())
    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for c, i in label2id.items()}
    df = df.copy()
    df["label_id"] = df[label_col].map(label2id).astype(int)
    return df, id2label, label2id


# ---------------------------------------------------------------------------
# BEE helpers
# ---------------------------------------------------------------------------

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalise embeddings to unit length (Euclidean norm)."""
    return embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8)


@torch.no_grad()
def embed_text_list(text_list: Sequence[str], model, tokenizer,
                    batch_size: int = 64, max_length: int = 128) -> np.ndarray:
    """
    Embed a list of keywords with the model's last-layer hidden state at the
    last (non-pad) token  ->  e_u^(m).
    """
    model.eval()
    all_embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch_texts = list(text_list[i: i + batch_size])
        inputs = tokenizer(batch_texts, padding=True, truncation=True,
                           max_length=max_length, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        if inputs["input_ids"].shape[1] == 0:
            continue
        outputs = model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        mask = inputs["attention_mask"]
        seq_lens = (mask.sum(dim=1) - 1).clamp(min=0)
        emb = last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), seq_lens]
        all_embeddings.append(emb.float().cpu().numpy())
    return np.vstack(all_embeddings)


def bee_scores(classifier_weights_norm: np.ndarray, keyword_embeddings_norm: np.ndarray) -> np.ndarray:
    """
    BEE keyword-class correlation:

        C_{u,y} = cos(e_u, w_y) - min_{y'} cos(e_u, w_y')

    Args:
        classifier_weights_norm: (num_classes, hidden)  rows of model.score.weight, L2-normalised
        keyword_embeddings_norm: (num_keywords, hidden) L2-normalised keyword embeddings
    Returns:
        (num_classes, num_keywords) array of non-negative scores
    """
    raw_sims = classifier_weights_norm @ keyword_embeddings_norm.T   # (C, K)
    min_sim = raw_sims.min(axis=0, keepdims=True)                    # min over classes
    return raw_sims - min_sim


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two flattened descriptors (0 if degenerate)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# Class-weighted Trainer (Stage 2 of training)
# ---------------------------------------------------------------------------

def make_weighted_trainer_class():
    """Import transformers lazily so that evaluation-only scripts stay light."""
    from transformers import Trainer

    class WeightedTrainer(Trainer):
        def __init__(self, *args, class_weights=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            if self.class_weights is not None:
                loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            else:
                loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer


def filter_keywords(raw_keywords: List[str], label_names: Sequence[str]) -> List[str]:
    """Drop class-name keywords and empty tokens."""
    filter_words = {str(w).lower() for w in label_names}
    return [str(k).strip() for k in raw_keywords
            if str(k).lower() not in filter_words and len(str(k).strip()) > 0]
