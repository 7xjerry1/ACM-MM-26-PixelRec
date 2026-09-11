import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_DATASETS = ("beauty", "games", "toys")


def load_dataset_config(dataset):
    name = str(dataset).lower()
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset {dataset!r}; choose from {SUPPORTED_DATASETS}.")
    path = PROJECT_ROOT / "configs" / f"{name}.json"
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["config_path"] = str(path)
    return config


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_paths(dataset):
    config = load_dataset_config(dataset)
    root = PROJECT_ROOT / "data" / config["dataset"]
    return {
        "root": root,
        "sequence": root / config["files"]["sequence"],
        "metadata": root / config["files"]["metadata"],
        "catalog": root / config["files"]["catalog"],
    }


def verify_bundled_data(dataset):
    config = load_dataset_config(dataset)
    paths = dataset_paths(dataset)
    failures = []
    for key in ("sequence", "metadata", "catalog"):
        path = paths[key]
        if not path.is_file():
            failures.append(f"missing {key}: {path}")
            continue
        expected = config["sha256"].get(key)
        if expected and sha256_file(path) != expected:
            failures.append(f"checksum mismatch for {key}: {path}")
    if failures:
        raise ValueError("Bundled dataset validation failed: " + "; ".join(failures))
    return paths


def build_model_args(config, cache_path, device, overrides=None):
    values = dict(config["training"])
    if overrides:
        values.update({key: value for key, value in overrides.items() if value is not None})
    values.update(
        {
            "item_size": int(config["num_items"]) + 1,
            "cuda_condition": str(device).startswith("cuda"),
            "pixelrec_token_cache_path": str(cache_path),
            "pixelrec_token_dim": int(config["compression"]["bottleneck_dim"]),
            "pixelrec_token_cache_len": int(config["vlm"]["max_token_cache_len"]),
        }
    )
    return SimpleNamespace(**values)
