"""
Stage 1 — Train the model suite for one dataset group and one seed.

For every training configuration (full, N-1 leave-one-out, single):

  1. Causal-LM finetuning of EleutherAI/pythia-14m on the configuration's texts
     (3 epochs, bs 32, lr 2e-5, wd 0.01, max_length 128)                -> clm_<cfg>/
  2. Sequence-classification head initialised from the CLM checkpoint and
     trained with class-weighted cross-entropy (3 epochs, bs 32, lr 2e-5) -> clf_<cfg>/

The classifier checkpoints (clf_<cfg>/) are what the SCD stage consumes:
`model.score.weight` gives the class vectors w_y and the backbone gives e_u.

Usage:
    python pipeline/train_models.py --group medical --seed 42
    python pipeline/train_models.py --group nli --seed 0 --configs alpha_n single_italki
"""

from __future__ import annotations

import argparse
import gc
import shutil
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root

from idscd.config import TrainConfig, add_common_args, resolve
from idscd.common import (encode_labels, get_device, load_group_csv,
                    make_weighted_trainer_class, set_seed)


def _training_args(output_dir: Path, epochs: int, bs: int, lr: float, wd: float,
                   seed: int, tc: TrainConfig) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=bs,
        learning_rate=lr,
        weight_decay=wd,
        logging_steps=tc.logging_steps,
        seed=seed,
        data_seed=seed,
        report_to="none",
        save_strategy="no",
    )


def train_clm(df, text_col: str, out_dir: Path, seed: int, tc: TrainConfig, device: str) -> None:
    """Stage 1: causal-LM finetuning."""
    tokenizer = AutoTokenizer.from_pretrained(tc.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(tc.model_id, torch_dtype=torch.float32).to(device)
    if any(torch.isnan(p).any() for p in model.parameters()):
        raise RuntimeError("Base model contains NaN parameters")

    ds = Dataset.from_pandas(df[[text_col]].rename(columns={text_col: "text"}), preserve_index=False)
    tok = ds.map(lambda ex: tokenizer(ex["text"], truncation=True, max_length=tc.max_length),
                 batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = _training_args(out_dir, tc.clm_epochs, tc.clm_batch_size, tc.clm_lr,
                          tc.clm_weight_decay, seed, tc)
    trainer = Trainer(model=model, args=args, train_dataset=tok, data_collator=collator)
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()


def train_classifier(df, text_col: str, label_col: str, clm_dir: Path, out_dir: Path,
                     seed: int, tc: TrainConfig, device: str) -> None:
    """Stage 2: class-weighted sequence-classification head on top of the CLM."""
    df, id2label, label2id = encode_labels(df, label_col)
    num_labels = len(id2label)

    tokenizer = AutoTokenizer.from_pretrained(clm_dir)
    tokenizer.pad_token = tokenizer.eos_token

    cw = compute_class_weight(class_weight="balanced",
                              classes=np.array(sorted(label2id, key=label2id.get)),
                              y=df[label_col].values)
    cw_tensor = torch.FloatTensor(cw).to(device)
    print("  class weights:", {id2label[i]: round(float(w), 4) for i, w in enumerate(cw)})

    model = AutoModelForSequenceClassification.from_pretrained(
        clm_dir, num_labels=num_labels, id2label=id2label, label2id=label2id,
        torch_dtype=torch.float32, ignore_mismatched_sizes=True)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    ds = Dataset.from_pandas(df[[text_col, "label_id"]]
                             .rename(columns={text_col: "text", "label_id": "labels"}),
                             preserve_index=False)
    tok = ds.map(lambda ex: tokenizer(ex["text"], truncation=True, max_length=tc.max_length), batched=True)
    tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # no weight decay for the classification stage (TrainingArguments default)
    args = _training_args(out_dir, tc.clf_epochs, tc.clf_batch_size, tc.clf_lr, 0.0, seed, tc)
    WeightedTrainer = make_weighted_trainer_class()
    trainer = WeightedTrainer(model=model, args=args, train_dataset=tok,
                              data_collator=collator, class_weights=cw_tensor)
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--configs", nargs="*", default=None,
                        help="subset of configuration keys to train (default: all, in the order "
                             "alpha_n, n-1_no_*, single_*)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip configurations whose clf_<cfg>/config.json already exists")
    parser.add_argument("--delete-clm", action="store_true",
                        help="delete the intermediate CLM checkpoint after stage 2 (saves disk)")
    parser.add_argument("--model-id", default=TrainConfig.model_id,
                        help="HF model id or local path of the base model (default EleutherAI/pythia-14m)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override the number of epochs of BOTH stages (default 3)")
    args = parser.parse_args()

    group, paths, _ = resolve(args)
    tc = TrainConfig(model_id=args.model_id)
    if args.epochs is not None:
        tc.clm_epochs = tc.clf_epochs = args.epochs
    device = get_device()
    set_seed(args.seed)
    print(f"group={group.name}  seed={args.seed}  device={device}\nmodels -> {paths.models_dir}")

    keys = args.configs or group.config_keys
    unknown = [k for k in keys if k not in group.config_keys]
    if unknown:
        raise SystemExit(f"Unknown config keys {unknown}; valid: {group.config_keys}")

    for key in keys:
        clf_dir = paths.clf_dir(key)
        clm_dir = paths.clm_dir(key)
        if args.skip_existing and (clf_dir / "config.json").exists():
            print(f"\n[skip] {key}: {clf_dir} exists")
            continue

        csv_path = group.csv_for_config(paths.data_dir, key)
        df = load_group_csv(group, csv_path)
        print(f"\n{'=' * 78}\nPIPELINE: {key}\n{'=' * 78}")
        print(f"  data: {csv_path} ({len(df):,} rows)")
        print(df[group.label_col].value_counts().to_string())

        print(f"\n--- Stage 1: CLM finetuning ({key}) ---")
        train_clm(df, group.text_col, clm_dir, args.seed, tc, device)

        print(f"\n--- Stage 2: classification head ({key}) ---")
        train_classifier(df, group.text_col, group.label_col, clm_dir, clf_dir, args.seed, tc, device)
        print(f"  saved classifier -> {clf_dir}")

        if args.delete_clm:
            shutil.rmtree(clm_dir, ignore_errors=True)

    print("\nDone. Trained configurations:")
    for key in keys:
        print(f"  {key:<22s} -> {paths.clf_dir(key)}")


if __name__ == "__main__":
    main()
