"""
Stage 4 — Experiments on the SCDs (final step: the Pearson(Δ_i, d^(j)) heatmap).

  (A) Leave-one-dataset-out diagnostic:
        Δ_i          = d_SCD^(all) - d_SCD^(-i)          ("contribution" of dataset i)
        S_Δ(i, j)    = Pearson(Δ_i, d_SCD^(j))
      -> contribution_i vs single_j matrix, diagonal-dominance check, ROC-AUC of
         diagonal (positives) vs off-diagonal (negatives), and the heatmap
         contrib_vs_single_pearson.{png,pdf} (rows d_SCD^(j), columns Δ_i)
      -> per-class variant of the same matrix (one heatmap per class + average)

  (B) idSCD membership score:
        s_SCD(i) = Pearson(d_SCD^(any), d_SCD^(i))
      members     : m_any = m_all   (trained with D_i)
      non-members : m_any = m_-i    (trained without D_i)
      flat + per-class scores pooled -> ROC-AUC.

Usage:
    python pipeline/evaluate.py --group medical --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root

from idscd.config import GroupConfig, Paths, add_common_args, resolve
from idscd.common import pearson


# ---------------------------------------------------------------------------
# (A) contribution_i vs single_j: Pearson(Δ_i, d^(j))
# ---------------------------------------------------------------------------

def delta_descriptors(mats: Dict[str, np.ndarray], group: GroupConfig) -> List[np.ndarray]:
    """Δ_i = d^(all) - d^(-i), flattened."""
    alpha_n = mats["alpha_n"]
    return [(alpha_n - mats[n1]).ravel() for n1 in group.n1_keys]


def s_delta_matrix(mats: Dict[str, np.ndarray], group: GroupConfig) -> np.ndarray:
    """S_Δ[i, j] = Pearson(Δ_i, d^(j))  (rows: contribution i, cols: single j)."""
    deltas = delta_descriptors(mats, group)
    singles = [mats[s].ravel() for s in group.single_keys]
    n = len(group.dataset_names)
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            S[i, j] = pearson(deltas[i], singles[j])
    return S


def s_delta_per_class(mats: Dict[str, np.ndarray], group: GroupConfig) -> Dict[str, np.ndarray]:
    """Same as s_delta_matrix but restricted to one class column at a time."""
    alpha_n = mats["alpha_n"]
    n = len(group.dataset_names)
    out = {}
    for c_idx, c_name in enumerate(group.labels):
        P = np.zeros((n, n))
        for i, n1 in enumerate(group.n1_keys):
            contrib_col = alpha_n[:, c_idx] - mats[n1][:, c_idx]
            for j, s in enumerate(group.single_keys):
                P[i, j] = pearson(contrib_col, mats[s][:, c_idx])
        out[c_name] = P
    return out


def diag_vs_offdiag_auc(S: np.ndarray) -> float:
    n = S.shape[0]
    mask = np.eye(n, dtype=bool)
    scores = np.concatenate([S[mask], S[~mask]])
    labels = np.concatenate([np.ones(n), np.zeros(n * n - n)])
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(auc(fpr, tpr))


def report_matrix(S: np.ndarray, names: List[str], title: str) -> None:
    n = len(names)
    df = pd.DataFrame(S, index=[f"Contrib {d}" for d in names], columns=[f"Single {d}" for d in names])
    print(f"\n{title}")
    print("(Diagonal = self-match, should be highest per row)")
    print(df.to_string(float_format="{:.4f}".format))
    hits = sum(int(np.argmax(S[i]) == i) for i in range(n))
    off = S[~np.eye(n, dtype=bool)]
    print(f"Diagonal-dominant rows: {hits}/{n}   diag mean {np.mean(np.diag(S)):.4f}   "
          f"off-diag mean {off.mean():.4f}   ROC-AUC(diag vs off-diag) = {diag_vs_offdiag_auc(S):.4f}")


def _tex(name: str) -> str:
    return name.replace("_", r"\_")


def plot_s_delta_heatmap(S: np.ndarray, names: List[str], out_stem: Path) -> None:
    """
    Heatmap of Pearson(Δ_i, d_SCD^(j)), transposed so that
    rows    = standalone descriptors d_SCD^(j)
    columns = leave-one-out changes Δ_i.
    """
    plt.rcParams["text.usetex"] = False
    x_labels = [rf"$\Delta_{{\mathrm{{{_tex(n)}}}}}$" for n in names]
    y_labels = [rf"$d_{{\mathrm{{SCD}}}}^{{(\mathrm{{{_tex(n)}}})}}$" for n in names]

    # cell width grows with the longest dataset name so the Δ labels don't collide
    n = len(names)
    longest = max(len(x) for x in names)
    cell_w = max(1.4, 0.19 * longest + 0.4)
    fig, ax = plt.subplots(figsize=(max(7, cell_w * n), max(6, 1.25 * n)))
    sns.heatmap(S.T, annot=True, fmt=".3f", cmap="coolwarm",
                xticklabels=x_labels, yticklabels=y_labels,
                center=0, ax=ax, linewidths=0.5, linecolor="white", cbar=False,
                annot_kws={"size": 20})
    ax.tick_params(labelsize=20)
    ax.set_yticklabels(y_labels, fontsize=20, rotation=0)
    ax.set_xticklabels(x_labels, fontsize=25)
    plt.tight_layout()
    fig.savefig(out_stem.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap -> {out_stem.with_suffix('.png')}  (+ .pdf)")


def plot_per_class_heatmaps(per_class: Dict[str, np.ndarray], avg: np.ndarray,
                            names: List[str], out_path: Path) -> None:
    panels = list(per_class.items()) + [("AVERAGE", avg)]
    n_cols = 3 if len(panels) <= 6 else 4
    n_rows = int(np.ceil(len(panels) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.2 * n_rows), squeeze=False)
    for idx, (c_name, P) in enumerate(panels):
        ax = axes[idx // n_cols][idx % n_cols]
        sns.heatmap(P, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1,
                    xticklabels=names, yticklabels=names, ax=ax, linewidths=0.5, linecolor="white")
        ax.set_title(c_name, fontsize=11, fontweight="bold")
    for idx in range(len(panels), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")
    fig.suptitle("Per-Class Pearson: contribution_i vs single_j (Zero-Fill BEE)",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved per-class heatmaps -> {out_path}")


# ---------------------------------------------------------------------------
# (B) idSCD membership score: Pearson(d^(any), d^(i))
# ---------------------------------------------------------------------------

def idscd_membership_scores(mats: Dict[str, np.ndarray], group: GroupConfig
                            ) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Returns (member_flat, nonmember_flat, member_cls, nonmember_cls):
      flat : Pearson over the whole descriptor
      cls  : Pearson over each class column separately (skipped if the single
             descriptor's column is constant)
    """
    alpha_n = mats["alpha_n"]
    member_flat, nonmember_flat, member_cls, nonmember_cls = [], [], [], []
    for n1, s in zip(group.n1_keys, group.single_keys):
        single, n1_mat = mats[s], mats[n1]
        member_flat.append(pearson(alpha_n, single))
        nonmember_flat.append(pearson(n1_mat, single))
        for c in range(len(group.labels)):
            if np.std(single[:, c]) < 1e-10:
                continue
            member_cls.append(pearson(alpha_n[:, c], single[:, c]))
            nonmember_cls.append(pearson(n1_mat[:, c], single[:, c]))
    return member_flat, nonmember_flat, member_cls, nonmember_cls


def roc_from_scores(members: List[float], nonmembers: List[float]):
    scores = np.array(members + nonmembers)
    labels = np.array([1] * len(members) + [0] * len(nonmembers))
    fpr, tpr, _ = roc_curve(labels, scores)
    return fpr, tpr, float(auc(fpr, tpr))


def plot_membership(fpr, tpr, roc_auc, member_flat, nonmember_flat, names, group_name, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].plot(fpr, tpr, color="darkorange", lw=2, label=f"Zero-Fill BEE (AUC = {roc_auc:.4f})")
    axes[0].plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random (0.5)")
    axes[0].set_xlim([-0.02, 1.02]); axes[0].set_ylim([-0.02, 1.05])
    axes[0].set_xlabel("False Positive Rate", fontsize=12)
    axes[0].set_ylabel("True Positive Rate", fontsize=12)
    axes[0].set_title(f"ROC — Leave-One-Out BEE MIA\n(Zero-Fill, Pearson, {group_name})",
                      fontsize=14, fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=11); axes[0].grid(True, alpha=0.3)

    x, w = np.arange(len(names)), 0.35
    axes[1].bar(x - w / 2, member_flat, w, label="Member (alpha_n)", color="steelblue")
    axes[1].bar(x + w / 2, nonmember_flat, w, label="Non-member (N-1)", color="salmon")
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=20, ha="right")
    axes[1].set_ylabel("Pearson Correlation to single_i", fontsize=11)
    axes[1].set_title("Member vs Non-member BEE Similarity\n(Zero-Fill, Pearson, per dataset)",
                      fontsize=13, fontweight="bold")
    axes[1].legend(fontsize=10); axes[1].grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved membership ROC -> {out_path}")


# ---------------------------------------------------------------------------

def load_aligned(paths: Paths, kw) -> Dict[str, np.ndarray]:
    cache = paths.aligned_cache(kw)
    if not cache.exists():
        raise FileNotFoundError(f"{cache} not found — run compute_scd.py first")
    data = np.load(cache)
    return {k: data[k] for k in data.files}


def run_evaluation(group: GroupConfig, paths: Paths, kw) -> np.ndarray:
    mats = load_aligned(paths, kw)
    names = group.dataset_names
    res = paths.results_dir
    print(f"group={group.name} seed={paths.seed}  |U|x|Y| = {mats['alpha_n'].shape}  results -> {res}")

    # ---- (B) idSCD membership ROC-AUC -------------------------------------
    m_flat, nm_flat, m_cls, nm_cls = idscd_membership_scores(mats, group)
    print("\nLeave-One-Out MIA — Zero-Fill BEE Pearson Correlation")
    print(f"{'Dataset':<16} {'Member':>10} {'Non-member':>12} {'Delta':>8} {'Hit':>5}")
    for d, m, nm in zip(names, m_flat, nm_flat):
        print(f"{d:<16} {m:>10.4f} {nm:>12.4f} {m - nm:>+8.4f} {'Y' if m > nm else '':>5}")
    fpr, tpr, roc_auc = roc_from_scores(m_flat + m_cls, nm_flat + nm_cls)
    print(f"Flat accuracy {sum(m > n for m, n in zip(m_flat, nm_flat))}/{len(names)}   "
          f"Per-class accuracy {sum(m > n for m, n in zip(m_cls, nm_cls))}/{len(m_cls)}   "
          f"AUC = {roc_auc:.4f}")
    np.savez(res / "bee_loo_scores.npz", members=np.array(m_flat + m_cls),
             nonmembers=np.array(nm_flat + nm_cls), fpr=fpr, tpr=tpr, auc=roc_auc)
    plot_membership(fpr, tpr, roc_auc, m_flat, nm_flat, names, group.name, res / "BEE_MIA_LOO_ROC_AUC_pearson.png")

    # ---- (A) per-class contribution_i vs single_j --------------------------
    per_class = s_delta_per_class(mats, group)
    avg = np.mean([per_class[c] for c in group.labels], axis=0)
    for c_name, P in per_class.items():
        print(f"\n{c_name}:")
        print(pd.DataFrame(P, index=names, columns=names).to_string(float_format="{:.4f}".format))
    report_matrix(avg, names, "Average Pearson(contribution_i, single_j) across all classes")
    plot_per_class_heatmaps(per_class, avg, names, res / "BEE_per_class_pearson_contribution_vs_single.png")

    # ---- (A) final result: Pearson(Δ_i, d^(j)) heatmap ---------------------
    S = s_delta_matrix(mats, group)
    report_matrix(S, names, "Contribution_i vs Single_j — Pearson Correlation (Zero-Fill)")
    # saved in the same orientation as the heatmap: rows d_SCD^(j), columns Δ_i
    pd.DataFrame(S.T, index=[f"d_{d}" for d in names], columns=[f"delta_{d}" for d in names]) \
        .to_csv(res / "contrib_vs_single_pearson.csv", float_format="%.6f")
    with open(res / "summary.txt", "w") as f:
        f.write(f"group={group.name} seed={paths.seed} top_k={kw.top_k} ngram={kw.ngram}\n")
        f.write(f"idSCD membership ROC-AUC: {roc_auc:.4f}\n")
        f.write(f"contrib-vs-single diag-vs-offdiag ROC-AUC: {diag_vs_offdiag_auc(S):.4f}\n")
        f.write(f"diagonal-dominant rows: {sum(int(np.argmax(S[i]) == i) for i in range(len(names)))}/{len(names)}\n")
    plot_s_delta_heatmap(S, names, res / "contrib_vs_single_pearson")
    return S


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    args = parser.parse_args()
    group, paths, kw = resolve(args)
    run_evaluation(group, paths, kw)


if __name__ == "__main__":
    main()
