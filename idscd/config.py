"""
Central configuration for the idSCD pipeline.

Directory layout, relative to the repository root (the folder containing this
file; override with --root if you want the data/models elsewhere):

    data/<group>/                    input CSVs (one per training configuration)
    models/<group>/seed_<s>/         trained checkpoints  (clm_<cfg>/, clf_<cfg>/)
    cache/<group>/                   YAKE keyword cache (seed independent)
    cache/<group>/seed_<s>/          aligned SCD matrices (.npz)
    results/<group>/seed_<s>/        heatmaps, ROC curves, CSV tables

Training-configuration keys:

    "alpha_n"            -> full model  m_all      (trained on D_all)
    "n-1_no_<dataset>"   -> leave-one-out model m_-i (trained on D_all \\ D_i)
    "single_<dataset>"   -> single-dataset model m_i  (trained on D_i)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------

MODEL_ID = "EleutherAI/pythia-14m"
DEFAULT_SEED = 42
MULTI_SEEDS = [0, 1, 2, 3]


@dataclass
class TrainConfig:
    model_id: str = MODEL_ID
    max_length: int = 128
    # Stage 1: causal-LM finetuning
    clm_epochs: int = 3
    clm_batch_size: int = 32
    clm_lr: float = 2e-5
    clm_weight_decay: float = 0.01
    # Stage 2: sequence-classification head (class-weighted CE)
    clf_epochs: int = 3
    clf_batch_size: int = 32
    clf_lr: float = 2e-5
    logging_steps: int = 100


@dataclass
class KeywordConfig:
    """YAKE settings."""
    top_k: int = 5000
    ngram: int = 3
    dedup_lim: float = 0.9
    language: str = "en"
    embed_batch_size: int = 64


# ---------------------------------------------------------------------------
# Dataset groups
# ---------------------------------------------------------------------------

@dataclass
class GroupConfig:
    name: str
    dataset_names: List[str]           # D_1 ... D_N (order used in the heatmaps)
    labels: List[str]                  # canonical label order (alphabetical)
    text_col: str = "text"
    label_col: str = "label"
    # Column used for YAKE keyword extraction (None -> same as text_col).
    # The PubMed CSVs keep the original sentence column 'abstract_text' (same
    # sentence as 'text'); YAKE reads it, training reads 'text'.
    yake_text_col: Optional[str] = None
    # Alternative column names that are normalised to text_col / label_col.
    text_col_aliases: List[str] = field(default_factory=list)
    label_col_aliases: List[str] = field(default_factory=list)
    # CSV file names inside data/<group>/
    full_csv: str = "full.csv"
    single_csvs: Dict[str, str] = field(default_factory=dict)   # dataset -> file
    n1_csvs: Dict[str, str] = field(default_factory=dict)       # dataset -> file

    # ----- derived helpers -------------------------------------------------
    @property
    def n1_keys(self) -> List[str]:
        return [f"n-1_no_{d}" for d in self.dataset_names]

    @property
    def single_keys(self) -> List[str]:
        return [f"single_{d}" for d in self.dataset_names]

    @property
    def config_keys(self) -> List[str]:
        return ["alpha_n"] + self.n1_keys + self.single_keys

    def csv_for_config(self, data_dir: Path, key: str) -> Path:
        if key == "alpha_n":
            return data_dir / self.full_csv
        if key.startswith("n-1_no_"):
            return data_dir / self.n1_csvs[key[len("n-1_no_"):]]
        if key.startswith("single_"):
            return data_dir / self.single_csvs[key[len("single_"):]]
        raise KeyError(key)

    @property
    def effective_yake_col(self) -> str:
        return self.yake_text_col or self.text_col


EMOTIONS = GroupConfig(
    name="emotions",
    dataset_names=["EmoLit", "Emotion20", "GoEmotions", "XED_SMED"],
    labels=["Angry", "Anticipation", "Disgust", "Fear", "Happy",
            "Neutral", "Sad", "Surprise", "Trust"],
    text_col="text",
    label_col="label",
    full_csv="all_4_datasets.csv",
    single_csvs={
        "EmoLit":     "EmoLit_mapped.csv",
        "Emotion20":  "EmotionDataset20_mapped.csv",
        "GoEmotions": "GoEmotions_mapped.csv",
        "XED_SMED":   "XED_and_SMED.csv",
    },
    n1_csvs={
        "EmoLit":     "no_EmoLit.csv",
        "Emotion20":  "no_EmotionDataset20.csv",
        "GoEmotions": "no_GoEmotions.csv",
        "XED_SMED":   "no_XED_SMED.csv",
    },
)

NLI = GroupConfig(
    name="nli",
    dataset_names=["reddit_l2", "efcamdat", "italki"],
    labels=["French", "German", "Italian", "Spanish", "Turkish"],
    text_col="text",
    label_col="l1",
    full_csv="balanced_corpus.csv",
    single_csvs={
        "reddit_l2": "reddit_l2_sub.csv",
        "efcamdat":  "efcamdat_sub.csv",
        "italki":    "italki_sub.csv",
    },
    n1_csvs={
        "reddit_l2": "all_without_reddit_l2.csv",
        "efcamdat":  "all_without_efcamdat.csv",
        "italki":    "all_without_italki.csv",
    },
)

MEDICAL = GroupConfig(
    name="medical",
    dataset_names=[f"split_{i}" for i in range(5)],
    labels=["BACKGROUND", "CONCLUSIONS", "METHODS", "OBJECTIVE", "RESULTS"],
    text_col="text",
    label_col="label",
    yake_text_col="abstract_text",
    label_col_aliases=["label_str"],
    full_csv="pubmed_all_clusters_combined.csv",
    single_csvs={f"split_{i}": f"pubmed_cluster_{i}.csv" for i in range(5)},
    n1_csvs={f"split_{i}": f"pubmed_all_except_cluster_{i}.csv" for i in range(5)},
)

GROUPS: Dict[str, GroupConfig] = {g.name: g for g in (EMOTIONS, NLI, MEDICAL)}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

@dataclass
class Paths:
    root: Path
    group: str
    seed: int

    @property
    def data_dir(self) -> Path:
        return self.root / "data" / self.group

    @property
    def raw_dir(self) -> Path:
        """Downloaded / original source files (used by the build_<group>_dataset.py scripts)."""
        return self.root / "data" / self.group / "raw"

    @property
    def models_dir(self) -> Path:
        return self.root / "models" / self.group / f"seed_{self.seed}"

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache" / self.group

    @property
    def seed_cache_dir(self) -> Path:
        return self.cache_dir / f"seed_{self.seed}"

    @property
    def results_dir(self) -> Path:
        return self.root / "results" / self.group / f"seed_{self.seed}"

    def clm_dir(self, key: str) -> Path:
        return self.models_dir / f"clm_{key}"

    def clf_dir(self, key: str) -> Path:
        return self.models_dir / f"clf_{key}"

    def yake_cache(self, kw: KeywordConfig) -> Path:
        return self.cache_dir / f"yake_keywords_n_{kw.ngram}_top_{kw.top_k}.json"

    def aligned_cache(self, kw: KeywordConfig) -> Path:
        return self.seed_cache_dir / f"aligned_matrices_n_{kw.ngram}_top_{kw.top_k}.npz"

    def union_keywords_file(self, kw: KeywordConfig) -> Path:
        return self.cache_dir / f"union_keywords_n_{kw.ngram}_top_{kw.top_k}.json"

    def ensure(self) -> "Paths":
        for d in (self.data_dir, self.models_dir, self.cache_dir,
                  self.seed_cache_dir, self.results_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self


REPO_ROOT = Path(__file__).resolve().parents[1]


def default_root() -> Path:
    return REPO_ROOT


def add_common_args(parser):
    """CLI arguments shared by every stage script."""
    parser.add_argument("--group", required=True, choices=sorted(GROUPS),
                        help="dataset group: emotions | nli | medical")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"random seed (default {DEFAULT_SEED}; multi-seed runs: {MULTI_SEEDS})")
    parser.add_argument("--root", type=Path, default=None,
                        help="folder holding data/, models/, cache/ and results/ "
                             "(default: the repository folder)")
    parser.add_argument("--top-k", type=int, default=KeywordConfig.top_k,
                        help="number of YAKE keywords per configuration (default 5000)")
    parser.add_argument("--ngram", type=int, default=KeywordConfig.ngram,
                        help="max YAKE n-gram length (default 3)")
    return parser


def resolve(args):
    """Turn parsed CLI args into (GroupConfig, Paths, KeywordConfig)."""
    group = GROUPS[args.group]
    root = args.root or default_root()
    paths = Paths(root=Path(root), group=group.name, seed=args.seed).ensure()
    kw = KeywordConfig(top_k=args.top_k, ngram=args.ngram)
    return group, paths, kw
