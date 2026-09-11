import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from .config import build_model_args, dataset_paths, load_dataset_config, verify_bundled_data
from .data import load_sequences, load_token_cache
from .model import PixelRec


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def resolve_device(device):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return resolved


def load_checkpoint(model, checkpoint_path, device):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    state = {
        (
            "pixelrec_token_aggregator." + name[len("vlm_token_aggregator.") :]
            if name.startswith("vlm_token_aggregator.")
            else name
        ): value
        for name, value in state.items()
    }
    model.load_state_dict(state)
    return payload


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_runtime(dataset, cache_path, device="auto", overrides=None):
    config = load_dataset_config(dataset)
    paths = verify_bundled_data(dataset)
    sequences = load_sequences(paths["sequence"])
    if len(sequences) != int(config["num_users"]):
        raise ValueError(f"Sequence user count {len(sequences)} != configured {config['num_users']}.")
    max_item = max(max(sequence) for sequence in sequences)
    if max_item != int(config["num_items"]):
        raise ValueError(f"Maximum item ID {max_item} != configured {config['num_items']}.")
    resolved_device = resolve_device(device)
    model_args = build_model_args(config, cache_path, resolved_device, overrides=overrides)
    if resolved_device.type == "cpu":
        model_args.pixelrec_token_cache_device = "cpu"
        model_args.pixelrec_token_compute_dtype = "float32"
    load_token_cache(
        cache_path,
        item_size=model_args.item_size,
        token_len=model_args.pixelrec_token_cache_len,
        token_dim=model_args.pixelrec_token_dim,
    )
    set_seed(model_args.seed)
    model = PixelRec(model_args).to(resolved_device)
    return config, paths, sequences, model_args, model, resolved_device
