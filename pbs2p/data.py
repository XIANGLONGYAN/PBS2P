"""Calibration and perplexity datasets for PBS2P."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

from datasets import Dataset, load_from_disk
import numpy as np
import torch
from torch import Tensor
from transformers import AutoTokenizer, PreTrainedTokenizerBase


@dataclass
class EncodedDataset:
    input_ids: Tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if "llama" in model_name.lower() and tokenizer.vocab_size == 32000:
        if tokenizer.bos_token_id != 1:
            tokenizer.bos_token_id = 1
        if tokenizer.eos_token_id != 2:
            tokenizer.eos_token_id = 2
    return tokenizer


def _load_local(data_root: Path, dataset: str, split: str) -> Dataset:
    path = data_root / dataset / split
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python prepare_data.py` first "
            "or pass the correct --data-root."
        )
    return load_from_disk(str(path))


def _sample_windows(
    input_ids: Tensor,
    num_samples: int,
    sequence_length: int,
    seed: int,
) -> list[tuple[Tensor, Tensor]]:
    random.seed(seed)
    samples: list[tuple[Tensor, Tensor]] = []
    for _ in range(num_samples):
        start = random.randint(0, input_ids.shape[1] - sequence_length - 1)
        inputs = input_ids[:, start : start + sequence_length]
        targets = inputs.clone()
        targets[:, :-1] = -100
        samples.append((inputs, targets))
    return samples


def _wikitext(
    data_root: Path,
    tokenizer: PreTrainedTokenizerBase,
    num_samples: int,
    sequence_length: int,
    seed: int,
) -> tuple[list[tuple[Tensor, Tensor]], EncodedDataset]:
    train = _load_local(data_root, "wikitext", "traindata")
    test = _load_local(data_root, "wikitext", "testdata")
    train_ids = tokenizer(
        " ".join(train["text"]),
        return_tensors="pt",
    ).input_ids
    test_ids = tokenizer(
        "\n\n".join(test["text"]),
        return_tensors="pt",
    ).input_ids
    return (
        _sample_windows(train_ids, num_samples, sequence_length, seed),
        EncodedDataset(test_ids),
    )


def _ptb(
    data_root: Path,
    tokenizer: PreTrainedTokenizerBase,
    num_samples: int,
    sequence_length: int,
    seed: int,
) -> tuple[list[tuple[Tensor, Tensor]], EncodedDataset]:
    train = _load_local(data_root, "ptb", "traindata")
    test = _load_local(data_root, "ptb", "testdata")
    train_ids = tokenizer(
        " ".join(train["sentence"]),
        return_tensors="pt",
    ).input_ids
    test_ids = tokenizer(
        " ".join(test["sentence"]),
        return_tensors="pt",
    ).input_ids
    return (
        _sample_windows(train_ids, num_samples, sequence_length, seed),
        EncodedDataset(test_ids),
    )


def _c4(
    data_root: Path,
    tokenizer: PreTrainedTokenizerBase,
    num_samples: int,
    sequence_length: int,
    seed: int,
) -> tuple[list[tuple[Tensor, Tensor]], EncodedDataset]:
    train = _load_local(data_root, "c4", "traindata")
    validation = _load_local(data_root, "c4", "valdata")
    random.seed(seed)

    samples: list[tuple[Tensor, Tensor]] = []
    for _ in range(num_samples):
        while True:
            document = train[random.randint(0, len(train) - 1)]["text"]
            encoded = tokenizer(document, return_tensors="pt").input_ids
            if encoded.shape[1] > sequence_length:
                break
        start = random.randint(0, encoded.shape[1] - sequence_length - 1)
        inputs = encoded[:, start : start + sequence_length]
        targets = inputs.clone()
        targets[:, :-1] = -100
        samples.append((inputs, targets))

    validation_ids = tokenizer(
        " ".join(validation[:1100]["text"]),
        return_tensors="pt",
    ).input_ids[:, : 256 * sequence_length]
    return samples, EncodedDataset(validation_ids)


def get_loaders(
    dataset: str,
    model_name: str,
    data_root: str | Path,
    *,
    num_samples: int = 128,
    seed: int = 0,
    sequence_length: int = 2048,
) -> tuple[list[tuple[Tensor, Tensor]], EncodedDataset]:
    """Return calibration samples and encoded perplexity evaluation data."""

    tokenizer = get_tokenizer(model_name)
    root = Path(data_root).expanduser()
    loaders = {
        "wikitext2": _wikitext,
        "ptb": _ptb,
        "c4": _c4,
    }
    try:
        loader = loaders[dataset]
    except KeyError as error:
        raise ValueError(f"Unsupported dataset: {dataset}") from error
    return loader(root, tokenizer, num_samples, sequence_length, seed)
