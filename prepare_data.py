#!/usr/bin/env python3
"""Download and prepare the calibration/evaluation datasets used by PBS2P."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset, load_dataset


def save(dataset: Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(path))
    print(f"Saved {len(dataset):,} rows to {path}")


def prepare_wikitext(data_root: Path) -> None:
    train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    save(train, data_root / "wikitext" / "traindata")
    save(test, data_root / "wikitext" / "testdata")


def prepare_ptb(data_root: Path) -> None:
    train = load_dataset(
        "ptb_text_only",
        "penn_treebank",
        split="train",
        trust_remote_code=True,
    )
    test = load_dataset(
        "ptb_text_only",
        "penn_treebank",
        split="test",
        trust_remote_code=True,
    )
    save(train, data_root / "ptb" / "traindata")
    save(test, data_root / "ptb" / "testdata")


def prepare_c4(data_root: Path) -> None:
    train = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )
    validation = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )
    save(train, data_root / "c4" / "traindata")
    save(validation, data_root / "c4" / "valdata")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="./data", type=Path)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("wikitext", "ptb", "c4"),
        default=("wikitext", "ptb", "c4"),
    )
    args = parser.parse_args()

    preparers = {
        "wikitext": prepare_wikitext,
        "ptb": prepare_ptb,
        "c4": prepare_c4,
    }
    for dataset in args.datasets:
        preparers[dataset](args.data_root)


if __name__ == "__main__":
    main()
