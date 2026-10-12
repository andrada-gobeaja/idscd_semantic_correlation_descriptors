"""
Stage 2 — YAKE keyword extraction per training configuration + global vocabulary U.

  * for each configuration (full / N-1 / single) the training texts are joined
    into one document and YAKE extracts `top_k` keywords
    (n-gram <= 3, dedupLim = 0.9, language = en);
  * class-name keywords and empty tokens are filtered out;
  * the per-configuration lists are cached as JSON (seed independent: the
    keywords only depend on the data);
  * the global keyword pool U is the sorted union of all lists.

Usage:
    python pipeline/extract_keywords.py --group emotions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import yake

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root

from idscd.config import GroupConfig, KeywordConfig, add_common_args, resolve
from idscd.common import filter_keywords, load_group_csv


def extract_yake_keywords(texts: List[str], kw: KeywordConfig) -> List[str]:
    sample_text = " ".join(str(t) for t in texts)
    extractor = yake.KeywordExtractor(lan=kw.language, n=kw.ngram, dedupLim=kw.dedup_lim,
                                      top=kw.top_k, features=None)
    return [k[0] for k in extractor.extract_keywords(sample_text)]


def load_or_extract_keywords(group: GroupConfig, data_dir: Path, cache_path: Path,
                             kw: KeywordConfig, force: bool = False) -> Dict[str, List[str]]:
    if cache_path.exists() and not force:
        print(f"Loading YAKE keywords from cache: {cache_path}")
        with open(cache_path) as f:
            per_config = json.load(f)
        missing = [k for k in group.config_keys if k not in per_config]
        if not missing:
            for k, v in per_config.items():
                print(f"  {k:<22s}: {len(v)} keywords")
            return per_config
        print(f"  cache incomplete, missing {missing}; extracting those")
    else:
        per_config = {}

    yake_col = group.effective_yake_col
    for key in group.config_keys:
        if key in per_config and not force:
            continue
        csv_path = group.csv_for_config(data_dir, key)
        df = load_group_csv(group, csv_path, need_label=False)
        col = yake_col if yake_col in df.columns else group.text_col
        if col != yake_col:
            print(f"  [warn] {csv_path.name}: column '{yake_col}' missing, using '{col}' for YAKE")
        texts = df[col].dropna().astype(str).tolist()
        print(f"  {key:<22s}: running YAKE on {len(texts):,} rows ...", flush=True)
        raw = extract_yake_keywords(texts, kw)
        per_config[key] = filter_keywords(raw, group.labels)
        print(f"  {key:<22s}: {len(per_config[key])} keywords")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:   # save incrementally (YAKE is slow on big corpora)
            json.dump(per_config, f)

    print(f"Saved YAKE keywords to: {cache_path}")
    return per_config


def union_keywords(per_config: Dict[str, List[str]]) -> List[str]:
    return sorted({kw for kws in per_config.values() for kw in kws})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--force", action="store_true", help="ignore the cache and re-run YAKE")
    args = parser.parse_args()
    group, paths, kw = resolve(args)

    per_config = load_or_extract_keywords(group, paths.data_dir, paths.yake_cache(kw), kw, force=args.force)
    union = union_keywords(per_config)
    with open(paths.union_keywords_file(kw), "w") as f:
        json.dump(union, f)
    print(f"\nUnion keyword count |U| = {len(union)}  -> {paths.union_keywords_file(kw)}")
    print(f"Examples: {union[:10]}")


if __name__ == "__main__":
    main()
