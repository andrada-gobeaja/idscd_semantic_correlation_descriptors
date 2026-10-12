"""
Optional — Cross-seed stability heatmap.

Compares the leave-one-out change Δ_i computed from one reference seed
(SEED_X, default 42) with the standalone descriptors d_SCD^(j) of models
trained with other seeds (SEEDS_Y):

    M[(s, j), i] = Pearson(Δ_i^(seed_x), d_SCD^(j), seed s)

Rows are grouped by seed (separated by a black line), columns are Δ_i.
Requires the aligned-matrix cache of every seed involved
(run compute_scd.py --seed <s> for each); all caches share the same
vocabulary U because the YAKE keywords are seed independent.

Usage:
    python pipeline/cross_seed_heatmap.py --group emotions --seed-x 42 --seeds-y 42 0 1 2 3
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root

from idscd.config import DEFAULT_SEED, MULTI_SEEDS, Paths, add_common_args, resolve
from idscd.common import pearson


def _tex(name: str) -> str:
    return name.replace("_", r"\_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--seed-x", type=int, default=DEFAULT_SEED, help="seed providing Δ_i (default 42)")
    parser.add_argument("--seeds-y", type=int, nargs="+", default=[DEFAULT_SEED] + MULTI_SEEDS,
                        help="seeds providing the standalone descriptors d^(j)")
    args = parser.parse_args()
    group, paths, kw = resolve(args)

    def load(seed: int):
        p = Paths(root=paths.root, group=group.name, seed=seed).aligned_cache(kw)
        if not p.exists():
            raise FileNotFoundError(f"missing aligned cache for seed {seed}: {p}")
        d = np.load(p)
        return {k: d[k] for k in d.files}

    mats = {s: load(s) for s in sorted(set([args.seed_x] + args.seeds_y))}
    ref_shape = mats[args.seed_x]["alpha_n"].shape
    for s, m in mats.items():
        for k, v in m.items():
            assert v.shape == ref_shape, f"shape mismatch seed {s} / {k}: {v.shape} vs {ref_shape}"

    names, n = group.dataset_names, len(group.dataset_names)
    delta_x = [(mats[args.seed_x]["alpha_n"] - mats[args.seed_x][n1]).ravel() for n1 in group.n1_keys]

    M = np.zeros((len(args.seeds_y) * n, n))
    for r_s, s in enumerate(args.seeds_y):
        for j, s_key in enumerate(group.single_keys):
            d_j = mats[s][s_key].ravel()
            for i in range(n):
                M[r_s * n + j, i] = pearson(delta_x[i], d_j)

    for r_s, s in enumerate(args.seeds_y):
        block = M[r_s * n:(r_s + 1) * n, :]
        hits = sum(int(np.argmax(block[j]) == j) for j in range(n))
        print(f"seed {s}: diagonal-dominant rows {hits}/{n}   diag mean {np.mean(np.diag(block)):.3f}  "
              f"offdiag mean {np.mean(block[~np.eye(n, dtype=bool)]):.3f}")

    plt.rcParams["text.usetex"] = False
    x_labels = [rf"$\Delta_{{\mathrm{{{_tex(d)}}}}}^{{(s{args.seed_x})}}$" for d in names]
    y_labels = [rf"$d_{{\mathrm{{SCD}}}}^{{(\mathrm{{{_tex(d)}}}),\,s{s}}}$" for s in args.seeds_y for d in names]
    fig, ax = plt.subplots(figsize=(7, 1.2 * len(args.seeds_y) * n / 2 + 2))
    sns.heatmap(M, annot=True, fmt=".3f", cmap="coolwarm", center=0,
                xticklabels=x_labels, yticklabels=y_labels,
                linewidths=0.5, linecolor="white", cbar=False, annot_kws={"size": 11}, ax=ax)
    for r_s in range(1, len(args.seeds_y)):          # separators between seed blocks
        ax.axhline(r_s * n, color="black", linewidth=2)
    ax.tick_params(labelsize=12)
    plt.tight_layout()

    out_dir = paths.root / "results" / group.name
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"BEE_crossseed_delta{args.seed_x}_vs_singles_{group.name}"
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    np.savetxt(stem.with_suffix(".csv"), M, delimiter=",", fmt="%.6f")
    print(f"Saved -> {stem}.png / .pdf / .csv")


if __name__ == "__main__":
    main()
