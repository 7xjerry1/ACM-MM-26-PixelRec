import json
import random
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler


def load_sequences(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            values = [int(value) for value in line.split()]
            if values:
                rows.append(values)
    if not rows:
        raise ValueError(f"No sequences found in {path}.")
    has_user_prefix = all(len(row) >= 2 and row[0] == index for index, row in enumerate(rows))
    sequences = [row[1:] if has_user_prefix else row for row in rows]
    if any(not sequence for sequence in sequences):
        raise ValueError(f"Empty sequence found in {path}.")
    return sequences


def load_catalog(path):
    by_id = {}
    by_asin = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            item_id = int(row["item_id"])
            asin = str(row["asin"])
            if item_id in by_id or asin in by_asin:
                raise ValueError(f"Duplicate catalog entry at line {line_number} in {path}.")
            row["item_id"] = item_id
            row["asin"] = asin
            row["title"] = str(row.get("title") or "")
            by_id[item_id] = row
            by_asin[asin] = row
    return by_id, by_asin


def load_token_cache(path, item_size=None, token_len=None, token_dim=None):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Token cache must be a dictionary with `tokens` and `mask`.")
    tokens, mask = payload.get("tokens"), payload.get("mask")
    if not isinstance(tokens, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise ValueError("Token cache must contain tensor `tokens` and `mask` values.")
    if tokens.ndim != 3 or mask.ndim != 2 or tuple(tokens.shape[:2]) != tuple(mask.shape):
        raise ValueError(f"Invalid token cache shapes: tokens={tuple(tokens.shape)}, mask={tuple(mask.shape)}.")
    expected = (item_size, token_len, token_dim)
    actual = tuple(tokens.shape)
    for got, wanted, label in zip(actual, expected, ("item_size", "token_len", "token_dim")):
        if wanted is not None and got != wanted:
            raise ValueError(f"Token cache {label} is {got}, expected {wanted}.")
    return tokens.contiguous(), mask.bool().contiguous()


class SequenceDataset(Dataset):
    def __init__(self, sequences, max_length=50, split="train", item_size=None):
        self.max_length = int(max_length)
        self.split = split
        self.item_size = int(item_size or (max(max(row) for row in sequences) + 1))
        self.samples = []
        if split == "train":
            for user_id, sequence in enumerate(sequences):
                training = sequence[-(self.max_length + 2) : -2]
                self.samples.extend((user_id, training[: index + 1]) for index in range(len(training)))
        elif split == "valid":
            self.samples = [(user_id, sequence[:-1]) for user_id, sequence in enumerate(sequences)]
        elif split == "test":
            self.samples = list(enumerate(sequences))
        else:
            raise ValueError("split must be train, valid, or test.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        user_id, items = self.samples[index]
        input_ids, answer = items[:-1], items[-1]
        padding = [0] * max(self.max_length - len(input_ids), 0)
        input_ids = (padding + input_ids)[-self.max_length :]
        negative = 0
        if self.split == "train":
            seen = set(items)
            negative = random.randint(1, self.item_size - 1)
            while negative in seen:
                negative = random.randint(1, self.item_size - 1)
        return (
            torch.tensor(user_id, dtype=torch.long),
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
            torch.tensor(negative, dtype=torch.long),
        )


def make_dataloaders(sequences, item_size, batch_size, max_length, num_workers=8):
    datasets = {
        split: SequenceDataset(sequences, max_length=max_length, split=split, item_size=item_size)
        for split in ("train", "valid", "test")
    }
    return {
        "train": DataLoader(
            datasets["train"], sampler=RandomSampler(datasets["train"]), batch_size=batch_size,
            num_workers=num_workers,
        ),
        "valid": DataLoader(
            datasets["valid"], sampler=SequentialSampler(datasets["valid"]), batch_size=batch_size,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            datasets["test"], sampler=SequentialSampler(datasets["test"]), batch_size=batch_size,
            num_workers=num_workers,
        ),
    }


def rating_matrix(sequences, item_size, split):
    trim = -2 if split == "valid" else -1
    rows, columns = [], []
    for user_id, sequence in enumerate(sequences):
        history = sequence[:trim]
        rows.extend([user_id] * len(history))
        columns.extend(history)
    values = np.ones(len(rows), dtype=np.float32)
    return csr_matrix((values, (rows, columns)), shape=(len(sequences), item_size))


def normalize_history(history, by_asin, item_size, max_length=50):
    normalized = []
    for value in history:
        text = str(value)
        if text in by_asin:
            normalized.append(int(by_asin[text]["item_id"]))
        elif text.isdigit() and 0 < int(text) < item_size:
            normalized.append(int(text))
        else:
            raise ValueError(f"Unknown item ID or ASIN: {value!r}.")
    if not normalized:
        raise ValueError("History must contain at least one item.")
    return normalized[-int(max_length) :]


def pad_history(history, max_length=50):
    return ([0] * max(int(max_length) - len(history), 0) + list(history))[-int(max_length) :]
