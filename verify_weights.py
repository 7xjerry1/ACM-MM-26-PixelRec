import argparse
import hashlib
import json
from pathlib import Path

import torch

from pixelrec.config import load_dataset_config
from pixelrec.evaluation import evaluate


def verify_sha256_manifest(root, manifest_path):
    root = Path(root)
    checked = 0
    for line_number, line in enumerate(Path(manifest_path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            expected, relative = line.split(maxsplit=1)
        except ValueError as error:
            raise ValueError(f"Invalid SHA-256 manifest line {line_number}: {line!r}") from error
        path = root / relative.lstrip("* ")
        if not path.is_file():
            raise FileNotFoundError(f"Missing file listed by SHA-256 manifest: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest().lower() != expected.lower():
            raise ValueError(f"SHA-256 mismatch: {path}")
        checked += 1
    if checked == 0:
        raise ValueError(f"No file entries found in {manifest_path}.")
    return checked


def verify_vlm_directory(vlm_dir):
    vlm_dir = Path(vlm_dir)
    required = ("config.json", "tokenizer_config.json", "preprocessor_config.json")
    missing = [name for name in required if not (vlm_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete VLM directory {vlm_dir}; missing: {missing}")
    config = json.loads((vlm_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_vl":
        raise ValueError(f"Expected model_type=qwen3_vl, got {config.get('model_type')!r}.")
    text_config = config.get("text_config") or {}
    if int(text_config.get("hidden_size", -1)) != 4096:
        raise ValueError(f"Expected Qwen3-VL hidden_size=4096, got {text_config.get('hidden_size')!r}.")
    index_path = vlm_dir / "model.safetensors.index.json"
    single_weight = vlm_dir / "model.safetensors"
    shards = []
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shards = sorted(set((index.get("weight_map") or {}).values()))
        if not shards:
            raise ValueError(f"No weight shards listed in {index_path}.")
        missing_shards = [name for name in shards if not (vlm_dir / name).is_file()]
        if missing_shards:
            raise FileNotFoundError(f"Missing VLM weight shards: {missing_shards}")
    elif single_weight.is_file():
        shards = [single_weight.name]
    else:
        raise FileNotFoundError("Missing model.safetensors or model.safetensors.index.json.")
    return {"vlm_dir": str(vlm_dir), "hidden_size": 4096, "weight_files": shards}


def verify_pixelrec_metrics(dataset, cache, checkpoint, device, tolerance):
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(checkpoint_payload, dict) and checkpoint_payload.get("dataset") not in (None, dataset):
        raise ValueError(
            f"Checkpoint dataset is {checkpoint_payload.get('dataset')!r}, expected {dataset!r}."
        )
    result = evaluate(dataset, cache, checkpoint, device=device)
    expected = load_dataset_config(dataset)["expected_metrics"]
    actual = result["metrics"]
    differences = {metric: abs(float(actual[metric]) - float(value)) for metric, value in expected.items()}
    failed = {metric: difference for metric, difference in differences.items() if difference > tolerance}
    if failed:
        raise ValueError(
            f"PixelRec benchmark verification failed for {dataset}; differences above tolerance {tolerance}: {failed}"
        )
    return {"dataset": dataset, "expected": expected, "actual": actual, "tolerance": tolerance}


def parse_args():
    parser = argparse.ArgumentParser(description="Verify downloaded Qwen3-VL and PixelRec weights.")
    parser.add_argument("--vlm-dir")
    parser.add_argument("--sha256-manifest")
    parser.add_argument("--dataset", choices=("beauty", "games", "toys"))
    parser.add_argument("--cache")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tolerance", type=float, default=0.0005)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.vlm_dir and not args.dataset:
        raise SystemExit("Provide --vlm-dir, --dataset, or both.")
    results = {}
    if args.vlm_dir:
        results["vlm"] = verify_vlm_directory(args.vlm_dir)
        if args.sha256_manifest:
            results["vlm"]["sha256_files_checked"] = verify_sha256_manifest(
                args.vlm_dir, args.sha256_manifest
            )
    elif args.sha256_manifest:
        raise SystemExit("--sha256-manifest requires --vlm-dir.")
    if args.dataset:
        if not args.cache or not args.checkpoint:
            raise SystemExit("--dataset requires both --cache and --checkpoint.")
        results["pixelrec"] = verify_pixelrec_metrics(
            args.dataset, args.cache, args.checkpoint, args.device, args.tolerance
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
