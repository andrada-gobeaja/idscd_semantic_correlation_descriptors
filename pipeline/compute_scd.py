"""
Stage 3 — Semantic Correlation Descriptors (SCDs) with zero-fill alignment.

For every trained classifier m:

  1. w_y  = rows of model.score.weight (L2-normalised)              class vectors
  2. e_u  = last-token hidden state of each of the model's OWN keywords
  3. C_{u,y} = cos(e_u, w_y) - min_{y'} cos(e_u, w_y')             (BEE)
  4. rows are mapped to the canonical label order and zero-filled into the
     shared vocabulary U  ->  C^(m) in R^{|U| x |Y|}

d_SCD^(m) = vec(C^(m)).  The aligned matrices of all configurations are cached
in one .npz file (cache/<group>/seed_<s>/aligned_matrices_*.npz).

Usage:
    python pipeline/compute_scd.py --group medical --seed 42
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root

from idscd.config import GroupConfig, KeywordConfig, Paths, add_common_args, resolve
from idscd.common import bee_scores, embed_text_list, get_device, normalize_embeddings, set_seed
from extract_keywords import load_or_extract_keywords, union_keywords


def resolve_model_paths(group: GroupConfig, paths: Paths) -> Dict[str, Path]:
    model_paths, missing = {}, []
    for key in group.config_keys:
        d = paths.clf_dir(key)
        if (d / "config.json").exists():
            model_paths[key] = d
            print(f"  {key:<22s} -> {d}")
        else:
            missing.append((key, d))
    for k, d in missing:
        print(f"  {k:<22s} !! not found: {d}")
    if missing:
        raise FileNotFoundError(f"{len(missing)} classifier checkpoint(s) unresolved — run train_models.py first.")
    return model_paths


@torch.no_grad()
def scd_matrix_for_model(model_path: Path, own_keywords: List[str], canonical_labels: List[str],
                         keyword_to_idx: Dict[str, int], kw: KeywordConfig, device: str) -> np.ndarray:
    """Zero-filled BEE correlation matrix C^(m) of shape (|U|, |Y|) for one model."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()

    # class directions w_y from the sequence-classification head
    classifier_weights = model.score.weight.detach().float().cpu().numpy()
    classifier_weights_norm = normalize_embeddings(classifier_weights)

    # keyword embeddings e_u using only this model's OWN keywords
    kw_emb = embed_text_list(own_keywords, model, tokenizer, batch_size=kw.embed_batch_size, max_length=128)
    kw_emb_norm = normalize_embeddings(kw_emb)

    sc_scores = bee_scores(classifier_weights_norm, kw_emb_norm)   # (num_model_labels, num_own)

    # map model label order -> canonical label order
    own_bee = np.zeros((len(own_keywords), len(canonical_labels)))
    id2label = {int(i): str(l) for i, l in model.config.id2label.items()}
    if [id2label.get(i) for i in range(len(canonical_labels))] == canonical_labels:
        own_bee = sc_scores.T
    else:
        for model_idx, label_name in id2label.items():
            if label_name in canonical_labels:
                own_bee[:, canonical_labels.index(label_name)] = sc_scores[model_idx, :]
        absent = [l for l in canonical_labels if l not in id2label.values()]
        if absent:
            print(f"    classes absent from model (zero columns): {absent}")

    # zero-fill into the shared vocabulary
    aligned = np.zeros((len(keyword_to_idx), len(canonical_labels)))
    for local_idx, k in enumerate(own_keywords):
        if k in keyword_to_idx:
            aligned[keyword_to_idx[k], :] = own_bee[local_idx, :]

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return aligned


def load_or_compute_aligned(group: GroupConfig, paths: Paths, kw: KeywordConfig,
                            force: bool = False) -> Dict[str, np.ndarray]:
    cache_path = paths.aligned_cache(kw)
    if cache_path.exists() and not force:
        print(f"Loading aligned matrices from cache: {cache_path}")
        cached = np.load(cache_path)
        mats = {k: cached[k] for k in cached.files}
        for k, m in mats.items():
            print(f"  {k:<22s} shape={m.shape}  nonzero_rows={int(np.any(m != 0, axis=1).sum())}")
        return mats

    per_config = load_or_extract_keywords(group, paths.data_dir, paths.yake_cache(kw), kw)
    union = union_keywords(per_config)
    keyword_to_idx = {k: i for i, k in enumerate(union)}
    with open(paths.union_keywords_file(kw), "w") as f:
        json.dump(union, f)
    print(f"|U| = {len(union)} keywords, |Y| = {len(group.labels)} classes")

    print("\nResolving classifier checkpoints:")
    model_paths = resolve_model_paths(group, paths)
    device = get_device()

    mats: Dict[str, np.ndarray] = {}
    for key, mpath in model_paths.items():
        own = per_config[key]
        print(f"\n{'=' * 60}\nSCD for {key}  ({len(own)} own keywords)\n{'=' * 60}")
        mats[key] = scd_matrix_for_model(mpath, own, group.labels, keyword_to_idx, kw, device)
        nz = int(np.any(mats[key] != 0, axis=1).sum())
        print(f"  C^(m) shape {mats[key].shape}, non-zero rows {nz}/{len(union)}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **mats)
    print(f"\nSaved aligned matrices -> {cache_path}")
    return mats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--force", action="store_true", help="recompute even if the .npz cache exists")
    args = parser.parse_args()
    group, paths, kw = resolve(args)
    set_seed(args.seed)
    load_or_compute_aligned(group, paths, kw, force=args.force)


if __name__ == "__main__":
    main()
