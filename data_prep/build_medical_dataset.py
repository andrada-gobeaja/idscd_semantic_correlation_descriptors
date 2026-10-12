"""
Build the medical (PubMed 200k RCT) dataset group from scratch.

Pipeline
--------
1. Download PubMed 200k RCT (Dernoncourt & Lee, 2017) from
   https://github.com/Franck-Dernoncourt/pubmed-rct/tree/master/PubMed_200k_RCT
   (train.7z, dev.txt, test.txt) and convert the text files to CSV with columns
       abstract_id, line_id, abstract_text, line_number, total_lines, target
   where line_id = "<abstract_id>_<line_number>_<total_lines>".

2. Draw the 12% stratified sample (272,434 sentences) as 5 stratified splits
       pubmed_rct_split_{0..4}.csv  ->  pubmed_rct_all_splits_combined.csv
   Two modes:
     --sample manifest   (default) select the exact sentences listed in
                         manifests/pubmed_rct_stratified_sample.csv.gz.
                         Reproduces the original sample byte for byte.
     --sample regenerate re-run the original procedure: TF-IDF (2,000 features)
                         + MiniBatchKMeans (k=5) + constrained greedy assignment
                         ("Strategy B"), then a 12% per-class sample of each split
                         (groupby(label).sample(frac=0.12, random_state=42)).
                         The split assignment is reproduced exactly on CPU; the
                         original run used GPU K-Means (cuML), which assigns a few
                         sentences near the size caps differently, so the sample
                         comes out ~90% identical, not 100%.

3. Topic clustering of the sample:
     sentences are grouped back into abstracts (by abstract_id), TF-IDF
     (10,000 features, English stop words, min_df=5, max_df=0.95) on the
     abstracts, MiniBatchKMeans (k=5, random_state=42, batch_size=2048,
     n_init=10); every sentence gets its abstract's cluster.
   Writes, in <out-dir>:
       pubmed_cluster_{0..4}.csv              single-cluster datasets
       pubmed_all_except_cluster_{0..4}.csv   leave-one-cluster-out datasets
       pubmed_all_clusters_combined.csv       full dataset

4. Check the sizes against the expected values.

Expected sizes (sentences): combined 272,434 · clusters 21,100 / 31,791 / 33,251 /
160,516 / 25,776. Requires scikit-learn >= 1.5 and < 1.9 (1.9 changes the
MiniBatchKMeans result and therefore the clusters).

Usage (Colab or local):
    pip install pandas "scikit-learn>=1.5,<1.9" py7zr
    python data_prep/build_medical_dataset.py                       # exact sample, writes data/medical/
    python data_prep/build_medical_dataset.py --sample regenerate    # re-run the sampling procedure
    python data_prep/build_medical_dataset.py --out-dir /content/medical --work-dir /content/pubmed_raw
    python data_prep/build_medical_dataset.py --overwrite            # rebuild even if the files exist
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
GITHUB_RAW = "https://raw.githubusercontent.com/Franck-Dernoncourt/pubmed-rct/master/PubMed_200k_RCT/{}"
MANIFEST = Path(__file__).resolve().parent / "manifests" / "pubmed_rct_stratified_sample.csv.gz"

NUM_SPLITS = 5
NUM_CLUSTERS = 5
SEED = 42
SAMPLE_FRAC = 0.12

EXPECTED_SAMPLE_ROWS = 272_434
EXPECTED_SAMPLE_ABSTRACTS = 148_024
EXPECTED_CLUSTER_SENTENCES = [21_100, 31_791, 33_251, 160_516, 25_776]
EXPECTED_CLUSTER_ABSTRACTS = [10_448, 16_632, 16_956, 91_164, 12_824]

RAW_COLUMNS = ["abstract_id", "line_id", "abstract_text", "line_number", "total_lines", "target"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1. Download + convert
# ---------------------------------------------------------------------------

def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        log(f"  present: {dest.name}")
        return
    log(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "idscd-build-medical"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.rename(dest)


def extract_7z(archive: Path, out_dir: Path) -> None:
    try:
        import py7zr
        with py7zr.SevenZipFile(archive, mode="r") as z:
            z.extractall(path=out_dir)
    except ImportError:
        exe = shutil.which("7z") or shutil.which("7za")
        if exe is None:
            sys.exit("Need py7zr (pip install py7zr) or the 7z command to extract train.7z")
        subprocess.run([exe, "x", "-y", f"-o{out_dir}", str(archive)], check=True,
                       stdout=subprocess.DEVNULL)


def parse_rct_txt(path: Path) -> pd.DataFrame:
    """Convert the original '###<abstract_id>' / '<LABEL>\\t<sentence>' format to rows."""
    rows, abstract_id, buf = [], None, []

    def flush():
        n = len(buf)
        for i, (label, text) in enumerate(buf):
            rows.append((abstract_id, f"{abstract_id}_{i}_{n}", text, i, n, label))

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("###"):
                if abstract_id is not None:
                    flush()
                abstract_id, buf = int(line[3:]), []
            elif line.strip():
                label, text = line.split("\t", 1)
                buf.append((label, text))
    if abstract_id is not None:
        flush()
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def prepare_raw(work_dir: Path) -> dict:
    """Download + convert; returns {'train': csv_path, 'dev': ..., 'test': ...}."""
    work_dir.mkdir(parents=True, exist_ok=True)
    log("Step 1/4: PubMed 200k RCT from GitHub")
    download(GITHUB_RAW.format("dev.txt"), work_dir / "dev.txt")
    download(GITHUB_RAW.format("test.txt"), work_dir / "test.txt")
    download(GITHUB_RAW.format("train.7z"), work_dir / "train.7z")
    if not (work_dir / "train.txt").exists():
        log("  extracting train.7z")
        extract_7z(work_dir / "train.7z", work_dir)

    csvs = {}
    for split in ("train", "dev", "test"):
        csv = work_dir / f"{split}.csv"
        if not csv.exists():
            df = parse_rct_txt(work_dir / f"{split}.txt")
            df.to_csv(csv, index=False)
            log(f"  {split}: {len(df):,} sentences -> {csv.name}")
        csvs[split] = csv
    return csvs


def load_raw(csvs: dict) -> pd.DataFrame:
    """train + dev + test with a source_split column. pandas' default NA handling is kept
    on purpose: it drops the one sentence whose text is literally 'None', as the original
    run did (2,270,286 -> 2,270,285 sentences)."""
    parts = []
    for split in ("train", "dev", "test"):
        part = pd.read_csv(csvs[split])
        part["source_split"] = split
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    return df.dropna(subset=["abstract_text", "target"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2a. Stratified sample from the manifest (exact)
# ---------------------------------------------------------------------------

def sample_from_manifest(raw: pd.DataFrame) -> dict:
    if not MANIFEST.exists():
        sys.exit(f"Manifest not found: {MANIFEST}")
    manifest = pd.read_csv(MANIFEST)
    log(f"Step 2/4: selecting {len(manifest):,} sentences listed in {MANIFEST.name}")
    keyed = raw.set_index(["line_id", "source_split"])
    splits = {}
    for i in range(NUM_SPLITS):
        ids = manifest[manifest["split"] == i]
        idx = pd.MultiIndex.from_frame(ids[["line_id", "source_split"]])
        missing = idx.difference(keyed.index)
        if len(missing):
            sys.exit(f"{len(missing)} manifest sentences not found in the raw data, e.g. {list(missing[:3])}")
        part = keyed.loc[idx].reset_index()
        splits[i] = part[RAW_COLUMNS + ["source_split"]]
    return splits


# ---------------------------------------------------------------------------
# 2b. Stratified sample regenerated from scratch (Strategy B + 12% sample)
# ---------------------------------------------------------------------------

def strategy_b_splits(df: pd.DataFrame, text_col="abstract_text", label_col="target",
                      k=NUM_SPLITS, max_features=2000, n_init=20, seed=SEED) -> np.ndarray:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import euclidean_distances
    from sklearn.preprocessing import normalize

    log(f"  TF-IDF ({max_features} features) on {len(df):,} sentences")
    X = TfidfVectorizer(max_features=max_features, sublinear_tf=True, strip_accents="unicode",
                        analyzer="word", min_df=5, dtype="float32").fit_transform(df[text_col])
    X = normalize(X)

    log("  MiniBatchKMeans centroids")
    km = MiniBatchKMeans(n_clusters=k, init="k-means++", n_init=n_init, batch_size=4096,
                         max_iter=500, random_state=seed)
    km.fit(X)
    centroids = km.cluster_centers_

    log("  distances to centroids")
    dist = np.vstack([euclidean_distances(X[i:i + 20_000].toarray().astype("float32"), centroids)
                      .astype("float32") for i in range(0, X.shape[0], 20_000)])

    log("  constrained greedy assignment")
    n = len(df)
    classes = sorted(df[label_col].unique())
    class_counts = df[label_col].value_counts().to_dict()
    max_size = int(n / k * 1.05)
    max_class = {c: int(v / k * 1.20) for c, v in class_counts.items()}
    split = np.full(n, -1, dtype=int)
    sizes = np.zeros(k, dtype=int)
    in_split = {s: {c: 0 for c in classes} for s in range(k)}
    labels = df[label_col].to_numpy()
    for idx in np.argsort(dist.min(axis=1)):
        c = labels[idx]
        for s in np.argsort(dist[idx]):
            if sizes[s] < max_size and in_split[s][c] < max_class[c]:
                break
        else:
            s = int(np.argmin(sizes))
        split[idx] = s
        sizes[s] += 1
        in_split[s][c] += 1
    return split


def sample_regenerated(raw: pd.DataFrame, work_dir: Path) -> dict:
    log("Step 2/4: regenerating the stratified sample (Strategy B + 12% per-class sample)")
    assignment = strategy_b_splits(raw)
    full_dir = work_dir / "strategy_B_full"
    full_dir.mkdir(parents=True, exist_ok=True)
    splits = {}
    for i in range(NUM_SPLITS):
        path = full_dir / f"pubmed_rct_split_{i}.csv"
        raw[assignment == i].to_csv(path, index=False)
        full = pd.read_csv(path)
        sampled = (full.groupby("target", group_keys=False)
                   .sample(frac=SAMPLE_FRAC, random_state=SEED)
                   .reset_index(drop=True))
        splits[i] = sampled
        log(f"  split {i}: {len(full):,} -> {len(sampled):,} sentences")
    return splits


# ---------------------------------------------------------------------------
# 3. Topic clustering of the sample
# ---------------------------------------------------------------------------

def cluster_abstracts(combined_csv: Path, out_dir: Path):
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    log("Step 3/4: topic clustering (TF-IDF on abstracts + MiniBatchKMeans)")
    df = pd.read_csv(combined_csv)
    if "target" in df.columns and "label" not in df.columns:
        df = df.rename(columns={"target": "label"})
    df = df.dropna(subset=["abstract_text", "label"])
    df["text"] = df["abstract_text"].astype(str)

    abstract_df = df.groupby("abstract_id").agg(full_text=("text", " ".join)).reset_index()
    X = TfidfVectorizer(max_features=10000, stop_words="english", min_df=5,
                        max_df=0.95).fit_transform(abstract_df["full_text"])
    km = MiniBatchKMeans(n_clusters=NUM_CLUSTERS, random_state=SEED, batch_size=2048, n_init=10)
    abstract_df["cluster"] = km.fit_predict(X)
    df = df.merge(abstract_df[["abstract_id", "cluster"]], on="abstract_id")

    for i in range(NUM_CLUSTERS):
        df[df["cluster"] == i].to_csv(out_dir / f"pubmed_cluster_{i}.csv", index=False)
        df[df["cluster"] != i].to_csv(out_dir / f"pubmed_all_except_cluster_{i}.csv", index=False)
    df.to_csv(out_dir / "pubmed_all_clusters_combined.csv", index=False)
    return df, abstract_df


# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", choices=["manifest", "regenerate"], default="manifest",
                   help="how to obtain the 12%% stratified sample (default: manifest = exact)")
    p.add_argument("--root", type=Path, default=None,
                   help="output root (default: the repository folder); writes <root>/data/medical/")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="write the dataset CSVs here instead of <root>/data/medical/")
    p.add_argument("--work-dir", type=Path, default=None,
                   help="downloads and intermediate files (default: <out-dir>/raw)")
    p.add_argument("--overwrite", action="store_true",
                   help="rebuild even if the cluster CSVs already exist")
    args = p.parse_args()

    out_dir = args.out_dir or ((args.root or REPO_ROOT) / "data" / "medical")
    outputs = ([out_dir / f"pubmed_cluster_{i}.csv" for i in range(NUM_CLUSTERS)]
               + [out_dir / f"pubmed_all_except_cluster_{i}.csv" for i in range(NUM_CLUSTERS)]
               + [out_dir / "pubmed_all_clusters_combined.csv"])
    if not args.overwrite and all(f.exists() for f in outputs):
        print(f"Medical dataset already present in {out_dir} (use --overwrite to rebuild).")
        return

    import sklearn
    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    if not ((1, 5) <= (major, minor) < (1, 9)):
        print(f"WARNING: scikit-learn {sklearn.__version__} — the clusters were verified with 1.5–1.8; "
              f"1.9 gives different clusters. Install 'scikit-learn>=1.5,<1.9' for an exact rebuild.",
              flush=True)

    work_dir = args.work_dir or (out_dir / "raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw(prepare_raw(work_dir))
    log(f"  raw corpus: {len(raw):,} sentences "
        f"({raw['source_split'].value_counts().to_dict()})")

    splits = sample_from_manifest(raw) if args.sample == "manifest" else sample_regenerated(raw, work_dir)

    strat_dir = out_dir / "stratified_sample"
    strat_dir.mkdir(parents=True, exist_ok=True)
    for i, part in splits.items():
        part.to_csv(strat_dir / f"pubmed_rct_split_{i}.csv", index=False)
    combined = pd.concat([pd.read_csv(strat_dir / f"pubmed_rct_split_{i}.csv") for i in range(NUM_SPLITS)],
                         ignore_index=True)
    combined_csv = strat_dir / "pubmed_rct_all_splits_combined.csv"
    combined.to_csv(combined_csv, index=False)
    log(f"  stratified sample: {len(combined):,} sentences, {combined['abstract_id'].nunique():,} abstracts "
        f"-> {strat_dir}")

    df, abstract_df = cluster_abstracts(combined_csv, out_dir)

    log("Step 4/4: checks")
    sent = df["cluster"].value_counts().sort_index().tolist()
    abst = abstract_df["cluster"].value_counts().sort_index().tolist()
    checks = [
        ("sample sentences", len(combined), EXPECTED_SAMPLE_ROWS),
        ("sample abstracts", combined["abstract_id"].nunique(), EXPECTED_SAMPLE_ABSTRACTS),
        ("sentences per cluster", sent, EXPECTED_CLUSTER_SENTENCES),
        ("abstracts per cluster", abst, EXPECTED_CLUSTER_ABSTRACTS),
    ]
    ok = True
    for name, got, want in checks:
        good = got == want
        ok &= good
        print(f"  {'OK ' if good else 'DIFF'}  {name:<22} {got}" + ("" if good else f"   (expected {want})"))
    for i in range(NUM_CLUSTERS):
        print(f"       pubmed_cluster_{i}.csv {sent[i]:>8,}   pubmed_all_except_cluster_{i}.csv {len(df) - sent[i]:>8,}")
    print(f"\nOutputs in {out_dir}")
    if not ok:
        note = (" (expected with --sample regenerate: the original sample used GPU K-Means)"
                if args.sample == "regenerate" else "")
        print(f"Sizes differ from the original dataset{note}.")
        sys.exit(0 if args.sample == "regenerate" else 1)
    print("All sizes match the original medical dataset.")


if __name__ == "__main__":
    main()
