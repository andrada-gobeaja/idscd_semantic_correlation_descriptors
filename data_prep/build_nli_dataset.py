"""
Build the NLI dataset group's leave-one-dataset-out files.

The NLI corpora cannot be downloaded automatically yet. Place these files in
data/nli/ first:
    reddit_l2_sub.csv, efcamdat_sub.csv, italki_sub.csv   single-dataset files
    balanced_corpus.csv                                   full corpus, with a `source` column

This script writes all_without_<dataset>.csv = balanced_corpus.csv minus that
source (columns l1, text). Existing files are kept unless --overwrite is given.

Usage:
    python data_prep/build_nli_dataset.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root

from idscd.config import DEFAULT_SEED, NLI, Paths, default_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=None,
                        help="data root (default: the repository folder); reads/writes <root>/data/nli/")
    parser.add_argument("--overwrite", action="store_true", help="rebuild output CSVs even if they exist")
    args = parser.parse_args()

    group = NLI
    data_dir = Paths(root=Path(args.root or default_root()), group=group.name, seed=DEFAULT_SEED).data_dir
    full_path = data_dir / group.full_csv
    if not full_path.exists():
        raise SystemExit(f"Missing {full_path} — put the NLI CSVs (balanced_corpus.csv with a `source` "
                         f"column and the *_sub.csv files) in {data_dir} first.")
    corpus = pd.read_csv(full_path)
    print(f"Balanced corpus: {len(corpus):,} rows\n{corpus['source'].value_counts().to_string()}")

    print("\nLeave-one-dataset-out files:")
    for name in group.dataset_names:
        out = data_dir / group.n1_csvs[name]
        if out.exists() and not args.overwrite:
            print(f"  exists  {out.name}  (use --overwrite to rebuild)")
            continue
        n1 = corpus[corpus["source"] != name][[group.label_col, group.text_col]].copy()
        n1.to_csv(out, index=False)
        print(f"  wrote   {out.name:<35s} {len(n1):>9,} rows")

    print("\nTraining configurations:")
    for key in group.config_keys:
        p = group.csv_for_config(data_dir, key)
        print(f"  {key:<22s} {'OK ' if p.exists() else 'MISSING'}  {p}")


if __name__ == "__main__":
    main()
