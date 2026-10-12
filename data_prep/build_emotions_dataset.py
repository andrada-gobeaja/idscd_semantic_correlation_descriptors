"""
Build the emotions dataset group from the internet.

1. download   the 5 raw sources into data/emotions/raw/
                EmoLit                    Zenodo record 7883954
                GoEmotions                Kaggle  debarshichanda/goemotions
                                          (fallback: google-research GitHub, identical files)
                SMED                      Kaggle  gangulyamrita/social-media-emotion-dataset
                XED (English)             GitHub  Helsinki-NLP/XED  AnnotatedData/en-annotated.tsv
                EmotionDataset20          Hugging Face  shreyaspulle98/emotion-dataset-20-emotions
2. map        each source to the 9 shared classes
                EmoLit_mapped.csv, EmotionDataset20_mapped.csv,
                GoEmotions_mapped.csv, XED_and_SMED.csv
3. merge      all_4_datasets.csv   (cross-dataset text de-duplication, first occurrence kept)
4. leave-one-out  no_<dataset>.csv (other three datasets, text de-duplication)

Raw files that are already present are not downloaded again, so you can also place
them by hand (same layout as data/emotions/raw/). Existing output CSVs are kept
unless --overwrite is given.

Usage:
    python data_prep/build_emotions_dataset.py
    python data_prep/build_emotions_dataset.py --skip-download    # raw files placed by hand
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import urllib.request
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root

from idscd.config import DEFAULT_SEED, EMOTIONS, GroupConfig, Paths, default_root

# ===========================================================================
# Emotions — raw sources
# ===========================================================================

ZENODO_EMOLIT_RECORD = "https://zenodo.org/api/records/7883954"
KAGGLE_GOEMOTIONS = "debarshichanda/goemotions"
GITHUB_GOEMOTIONS = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/{}"
KAGGLE_SMED = "gangulyamrita/social-media-emotion-dataset"
GITHUB_XED_EN = "https://raw.githubusercontent.com/Helsinki-NLP/XED/master/AnnotatedData/en-annotated.tsv"
HF_EMOTION20 = "hf://datasets/shreyaspulle98/emotion-dataset-20-emotions/emotion_dataset_v5_clean.csv"

EMOLIT_FILES = ["gold.tsv", "trn.tsv", "tst.tsv", "val.tsv"]
GOEMOTIONS_FILES = ["train.tsv", "dev.tsv", "test.tsv", "emotions.txt"]


def raw_layout(raw: Path) -> Dict[str, Path]:
    return {
        "emolit_dir": raw / "EmoLit",
        "goemotions_dir": raw / "GoEmotions" / "data",
        "smed_csv": raw / "SMED" / "Social Media Emotion Dataset.csv",
        "xed_tsv": raw / "XED" / "en-annotated.tsv",
        "emotion20_csv": raw / "emotion-dataset-20-emotions" / "emotion_dataset_v5_clean.csv",
    }


def _http_get(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "idscd-build-emotions"})
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def _is_lfs_pointer(p: Path) -> bool:
    with open(p, "rb") as f:
        return f.read(40).startswith(b"version https://git-lfs")


def _kaggle_download(slug: str, dest_dir: Path) -> None:
    import kagglehub  # imported lazily: only needed for downloading
    src = Path(kagglehub.dataset_download(slug))
    print(f"    downloaded to cache: {src}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dest_dir, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dest_dir / src.name)


def download_emolit(target_dir: Path) -> None:
    """Fetch every file of the Zenodo record, unpack archives, keep the 4 TSV splits."""
    with urllib.request.urlopen(ZENODO_EMOLIT_RECORD, timeout=60) as r:
        record = json.load(r)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for entry in record.get("files", []):
            name = entry.get("key") or entry.get("filename")
            links = entry.get("links", {})
            url = links.get("content") or links.get("self") or links.get("download")
            print(f"    {name}")
            _http_get(url, tmp / name)
            if zipfile.is_zipfile(tmp / name):
                with zipfile.ZipFile(tmp / name) as z:
                    z.extractall(tmp / "unpacked")
            elif tarfile.is_tarfile(tmp / name):
                with tarfile.open(tmp / name) as t:
                    t.extractall(tmp / "unpacked")
        target_dir.mkdir(parents=True, exist_ok=True)
        for fname in EMOLIT_FILES:
            hits = sorted(tmp.rglob(fname))
            if not hits:
                found = sorted(p.name for p in tmp.rglob("*") if p.is_file())
                raise FileNotFoundError(f"EmoLit: {fname} not found in the Zenodo record; files: {found}")
            shutil.copy2(hits[0], target_dir / fname)


def download_emotions(raw: Path) -> None:
    L = raw_layout(raw)
    print(f"Raw files -> {raw}")

    # EmoLit (Zenodo)
    if all((L["emolit_dir"] / f).exists() and not _is_lfs_pointer(L["emolit_dir"] / f) for f in EMOLIT_FILES):
        print("  EmoLit: present")
    else:
        print(f"  EmoLit: downloading from {ZENODO_EMOLIT_RECORD}")
        download_emolit(L["emolit_dir"])

    # GoEmotions (Kaggle, GitHub fallback)
    if all((L["goemotions_dir"] / f).exists() for f in GOEMOTIONS_FILES):
        print("  GoEmotions: present")
    else:
        print(f"  GoEmotions: downloading Kaggle {KAGGLE_GOEMOTIONS}")
        try:
            _kaggle_download(KAGGLE_GOEMOTIONS, raw / "GoEmotions")
        except Exception as e:
            print(f"    Kaggle failed ({e}); using the identical google-research GitHub files")
            for f in GOEMOTIONS_FILES:
                _http_get(GITHUB_GOEMOTIONS.format(f), L["goemotions_dir"] / f)

    # SMED (Kaggle)
    if L["smed_csv"].exists():
        print("  SMED: present")
    else:
        print(f"  SMED: downloading Kaggle {KAGGLE_SMED}")
        try:
            _kaggle_download(KAGGLE_SMED, L["smed_csv"].parent)
        except Exception as e:
            raise SystemExit(
                f"SMED download failed ({e}).\nDownload it from "
                f"https://www.kaggle.com/datasets/{KAGGLE_SMED} (Kaggle credentials may be "
                f"required: ~/.kaggle/kaggle.json) and place it at:\n  {L['smed_csv']}")

    # XED English (GitHub) — normalise CRLF line endings to LF
    if L["xed_tsv"].exists():
        print("  XED: present")
    else:
        print(f"  XED: downloading {GITHUB_XED_EN}")
        _http_get(GITHUB_XED_EN, L["xed_tsv"])
        data = L["xed_tsv"].read_bytes().replace(b"\r\n", b"\n")
        L["xed_tsv"].write_bytes(data)

    # EmotionDataset20 (Hugging Face) — read with pandas defaults and re-saved, as originally done
    if L["emotion20_csv"].exists():
        print("  EmotionDataset20: present")
    else:
        print(f"  EmotionDataset20: downloading {HF_EMOTION20}")
        df = pd.read_csv(HF_EMOTION20)
        L["emotion20_csv"].parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(L["emotion20_csv"], index=False)


# ===========================================================================
# Emotions — label mapping to the 9 shared classes
# ===========================================================================

EMOLIT_MAPPING = {
    "joy": "Happy", "amusement": "Happy", "excitement": "Happy", "gratitude": "Happy",
    "love": "Happy", "pride": "Happy", "relief": "Happy", "admiration": "Happy",
    "nostalgia": "Sad", "sadness": "Sad", "despair": "Sad", "disappointment": "Sad",
    "grief": "Sad", "guilt": "Sad", "embarrassment": "Sad", "pain": "Sad",
    "anger": "Angry", "annoyance": "Angry", "frustration": "Angry", "envy": "Angry",
    "indifference": "Neutral", "boredom": "Neutral", "calmness": "Neutral",
    "fear": "Fear", "nervousness": "Fear", "doubt": "Fear",
    "disgust": "Disgust", "disapproval": "Disgust",
    "surprise": "Surprise",
    "curiosity": "Anticipation", "desire": "Anticipation", "optimism": "Anticipation", "greed": "Anticipation",
    "trust": "Trust", "faith": "Trust", "approval": "Trust", "caring": "Trust", "courage": "Trust",
}

GOEMOTIONS_MAPPING = {
    "admiration": "Happy", "amusement": "Happy", "anger": "Angry", "annoyance": "Angry",
    "approval": "Trust", "caring": "Trust", "confusion": "Surprise", "curiosity": "Anticipation",
    "desire": "Anticipation", "disappointment": "Sad", "disapproval": "Disgust", "disgust": "Disgust",
    "embarrassment": "Sad", "excitement": "Happy", "fear": "Fear", "gratitude": "Happy",
    "grief": "Sad", "joy": "Happy", "love": "Happy", "nervousness": "Fear",
    "optimism": "Anticipation", "pride": "Happy", "realization": "Surprise", "relief": "Happy",
    "remorse": "Sad", "sadness": "Sad", "surprise": "Surprise", "neutral": "Neutral",
}

# "love" and "gratitude" map to Trust (not Happy) for this dataset.
EMOTION20_MAPPING = {
    "happiness": "Happy", "excitement": "Happy", "pride": "Happy", "relief": "Happy",
    "love": "Trust", "gratitude": "Trust",
    "sadness": "Sad", "disappointment": "Sad", "guilt": "Sad", "embarrassment": "Sad", "loneliness": "Sad",
    "anger": "Angry", "frustration": "Angry", "jealousy": "Angry",
    "confusion": "Neutral",
    "hope": "Anticipation", "anxiety": "Anticipation",
    "disgust": "Disgust", "fear": "Fear", "surprise": "Surprise",
}

XED_MAPPING = {"1": "Angry", "2": "Anticipation", "3": "Disgust", "4": "Fear",
               "5": "Happy", "6": "Sad", "7": "Surprise", "8": "Trust"}


EMOLIT_NON_EMOTION_COLS = ("tid", "text", "split")


def map_emolit(emolit_dir: Path, chunksize: int = 50_000) -> pd.DataFrame:
    """
    All TSV splits; dedup on text; label = column with the highest score.

    The files are read in chunks and the score columns are converted to numbers
    chunk by chunk, so the whole corpus is never held as strings at once
    (that needs several GB of RAM). Same result as reading everything as text,
    concatenating, de-duplicating and converting afterwards: the conversion is
    per cell and de-duplication only looks at `text`.
    """
    tsvs = sorted(emolit_dir.glob("*.tsv"))
    for p in tsvs:
        if _is_lfs_pointer(p):
            raise SystemExit(f"{p} is a Git LFS pointer, not data — download the real file "
                             f"(Zenodo record 7883954) or delete it and rerun without --skip-download.")
    parts = []
    for p in tsvs:
        for chunk in pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, chunksize=chunksize):
            for c in chunk.columns:
                if c not in EMOLIT_NON_EMOTION_COLS:
                    chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
            parts.append(chunk)
    df = pd.concat(parts, ignore_index=True)
    del parts
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    emotion_cols = [c for c in df.columns if c not in EMOLIT_NON_EMOTION_COLS]
    df[emotion_cols] = df[emotion_cols].fillna(0)
    df["label"] = df[emotion_cols].idxmax(axis=1).map(EMOLIT_MAPPING)
    return df.dropna(subset=["label"])[["text", "label"]]


def map_goemotions(go_dir: Path) -> pd.DataFrame:
    """train+dev+test; strip + dedup on text; label = FIRST emotion id."""
    emotion_names = [l.strip() for l in open(go_dir / "emotions.txt").readlines()]
    df = pd.concat([pd.read_csv(go_dir / f, sep="\t", header=None,
                                names=["text", "emotion_ids", "comment_id"],
                                dtype=str, keep_default_na=False)
                    for f in ("train.tsv", "dev.tsv", "test.tsv")], ignore_index=True)
    df["text"] = df["text"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    def first_emotion(id_str):
        try:
            i = int(id_str.split(",")[0])
            return emotion_names[i] if 0 <= i < len(emotion_names) else None
        except (ValueError, IndexError):
            return None

    df["label"] = df["emotion_ids"].apply(first_emotion).map(GOEMOTIONS_MAPPING)
    return df.dropna(subset=["label"])[["text", "label"]]


def map_emotion20(csv: Path) -> pd.DataFrame:
    """strip + dedup on text; lower-cased label mapped; unmapped labels dropped."""
    df = pd.read_csv(csv, keep_default_na=False)
    if "sentence" in df.columns:
        df = df.rename(columns={"sentence": "text"})
    if "text" not in df.columns:
        df = df.rename(columns={df.columns[0]: "text"})
    if "emotion" not in df.columns:
        df = df.rename(columns={df.columns[1]: "emotion"})
    df["text"] = df["text"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df["label"] = df["emotion"].astype(str).str.strip().str.lower().map(EMOTION20_MAPPING)
    return df.dropna(subset=["label"])[["text", "label"]]


def map_xed_smed(smed_csv: Path, xed_tsv: Path) -> pd.DataFrame:
    """SMED (title-cased labels) + XED (one row per label id), dedup on text, SMED first."""
    smed = pd.read_csv(smed_csv, keep_default_na=False)
    if "label" not in smed.columns:
        smed = smed.rename(columns={smed.columns[0]: "text", smed.columns[1]: "label"})
    elif "text" not in smed.columns:
        smed = smed.rename(columns={smed.columns[0]: "text"})
    smed = smed[["text", "label"]].copy()
    smed["text"] = smed["text"].astype(str).str.strip()
    smed["label"] = smed["label"].astype(str).str.strip().str.title()

    xed = pd.read_csv(xed_tsv, sep="\t", header=None, names=["text", "label_ids"],
                      dtype=str, keep_default_na=False)
    xed["label_ids"] = xed["label_ids"].str.split(",")
    xed = xed.explode("label_ids")
    xed["label_ids"] = xed["label_ids"].str.strip()
    xed["label"] = xed["label_ids"].map(XED_MAPPING)
    xed = xed.dropna(subset=["label"])
    xed["text"] = xed["text"].astype(str).str.strip()
    xed = xed[["text", "label"]]

    merged = pd.concat([smed, xed], ignore_index=True)
    return merged.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)


# ===========================================================================
# Emotions — full corpus and leave-one-out files
# ===========================================================================

def merge_full(single_paths: List[Path]) -> pd.DataFrame:
    """Concatenate (in order), strip, cross-dataset dedup on text keeping the first occurrence."""
    df = pd.concat([pd.read_csv(p, keep_default_na=False) for p in single_paths], ignore_index=True)
    df["text"] = df["text"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    return df[["text", "label"]].copy()


def leave_one_out(single_paths: Dict[str, Path], exclude: str) -> pd.DataFrame:
    """Concatenate the other datasets, dedup on text keeping the first occurrence."""
    dfs = []
    for name, p in single_paths.items():
        if name == exclude:
            continue
        df = pd.read_csv(p).dropna(subset=["text", "label"])
        df["text"] = df["text"].astype(str)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["text"], keep="first")


def _write(df: pd.DataFrame, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        print(f"  exists  {path.name}  (use --overwrite to rebuild)")
        return
    df.to_csv(path, index=False)
    print(f"  wrote   {path.name:<35s} {len(df):>9,} rows")


def build_emotions(group: GroupConfig, raw: Path, data_dir: Path, overwrite: bool) -> None:
    L = raw_layout(raw)
    singles = {name: data_dir / group.single_csvs[name] for name in group.dataset_names}

    print("\nMapping sources to the 9 classes:")
    builders = {
        "EmoLit": lambda: map_emolit(L["emolit_dir"]),
        "Emotion20": lambda: map_emotion20(L["emotion20_csv"]),
        "GoEmotions": lambda: map_goemotions(L["goemotions_dir"]),
        "XED_SMED": lambda: map_xed_smed(L["smed_csv"], L["xed_tsv"]),
    }
    for name, fn in builders.items():
        if singles[name].exists() and not overwrite:
            print(f"  exists  {singles[name].name}  (use --overwrite to rebuild)")
            continue
        _write(fn(), singles[name], overwrite=True)

    print("\nFull corpus:")
    _write(merge_full([singles[n] for n in group.dataset_names]),
           data_dir / group.full_csv, overwrite)

    print("\nLeave-one-dataset-out files:")
    for name in group.dataset_names:
        _write(leave_one_out(singles, name), data_dir / group.n1_csvs[name], overwrite)


# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=None,
                        help="output root (default: the repository folder); writes <root>/data/emotions/")
    parser.add_argument("--skip-download", action="store_true",
                        help="do not download, use the raw files already in data/emotions/raw/")
    parser.add_argument("--overwrite", action="store_true", help="rebuild output CSVs even if they exist")
    args = parser.parse_args()

    group = EMOTIONS
    paths = Paths(root=Path(args.root or default_root()), group=group.name, seed=DEFAULT_SEED)
    paths.data_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        download_emotions(paths.raw_dir)
    build_emotions(group, paths.raw_dir, paths.data_dir, args.overwrite)

    print("\nTraining configurations:")
    for key in group.config_keys:
        p = group.csv_for_config(paths.data_dir, key)
        print(f"  {key:<22s} {'OK ' if p.exists() else 'MISSING'}  {p}")


if __name__ == "__main__":
    main()
