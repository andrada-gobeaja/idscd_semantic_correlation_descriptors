"""
End-to-end driver: data -> models -> keywords -> SCDs -> Pearson(Δ_i, d^(j)) heatmap.

Stages (any subset can be selected with --stages):
    data      build the dataset CSVs with build_<group>_dataset.py
              (emotions/medical: downloaded and rebuilt; nli: leave-one-out files
              from the CSVs you provide)
    train     finetune Pythia-14M CLM + class-weighted classifier per configuration
    keywords  YAKE top-k (n<=3) keywords per configuration and union vocabulary
    scd       zero-fill aligned BEE matrices (SCDs) for every trained model
    evaluate  contribution_i vs single_j Pearson matrix + heatmap, per-class variant,
              idSCD membership ROC-AUC

Examples:
    # whole pipeline for the medical group
    python run_pipeline.py --group medical --seed 42

    # multi-seed runs, then the cross-seed heatmap
    for s in 0 1 2 3; do python run_pipeline.py --group medical --seed $s --stages train scd evaluate; done
    python pipeline/cross_seed_heatmap.py --group medical --seed-x 42 --seeds-y 42 0 1 2 3

    # only re-run the analysis on cached SCDs
    python run_pipeline.py --group nli --seed 42 --stages evaluate
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from idscd.config import add_common_args, resolve

STAGES = ["data", "train", "keywords", "scd", "evaluate"]
SCRIPTS = {
    "data": "data_prep/build_{group}_dataset.py",
    "train": "pipeline/train_models.py",
    "keywords": "pipeline/extract_keywords.py",
    "scd": "pipeline/compute_scd.py",
    "evaluate": "pipeline/evaluate.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    parser.add_argument("--skip-existing", action="store_true",
                        help="train: skip configurations that already have a classifier checkpoint")
    parser.add_argument("--delete-clm", action="store_true", help="train: delete CLM checkpoints after stage 2")
    parser.add_argument("--skip-download", action="store_true",
                        help="data (emotions): use raw files already in data/emotions/raw/ instead of downloading")
    parser.add_argument("--model-id", default=None, help="train: base model id / local path")
    parser.add_argument("--epochs", type=int, default=None, help="train: override epochs (debugging)")
    args = parser.parse_args()
    group, paths, kw = resolve(args)

    here = Path(__file__).resolve().parent
    common = ["--group", group.name, "--seed", str(args.seed), "--root", str(paths.root),
              "--top-k", str(kw.top_k), "--ngram", str(kw.ngram)]
    train_extra = (["--skip-existing"] if args.skip_existing else []) \
        + (["--delete-clm"] if args.delete_clm else []) \
        + (["--model-id", args.model_id] if args.model_id else []) \
        + (["--epochs", str(args.epochs)] if args.epochs is not None else [])
    data_extra = ["--skip-download"] if (args.skip_download and group.name == "emotions") else []

    for stage in STAGES:
        if stage not in args.stages:
            continue
        if stage == "data":  # the dataset builders only take --root
            cmd = [sys.executable, str(here / SCRIPTS[stage].format(group=group.name)),
                   "--root", str(paths.root)] + data_extra
        else:
            cmd = [sys.executable, str(here / SCRIPTS[stage])] + common + (train_extra if stage == "train" else [])
        print(f"\n{'#' * 78}\n# STAGE {stage}: {' '.join(cmd)}\n{'#' * 78}", flush=True)
        subprocess.run(cmd, check=True)

    print(f"\nAll requested stages finished. Results: {paths.results_dir}")
    print(f"Final heatmap: {paths.results_dir / 'contrib_vs_single_pearson.png'}")


if __name__ == "__main__":
    main()
