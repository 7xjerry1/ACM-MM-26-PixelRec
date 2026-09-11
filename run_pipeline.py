import argparse
import json
import subprocess
import sys
from pathlib import Path

from pixelrec.config import PROJECT_ROOT, load_dataset_config
from pixelrec.data import load_token_cache


PIPELINE_STAGES = ("preprocess", "vlm", "compress", "train", "evaluate", "recommend")
ALL_STAGES = PIPELINE_STAGES[:5]


def run(command):
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], cwd=PROJECT_ROOT, check=True)


def cache_is_valid(path, item_size, token_len, token_dim):
    try:
        load_token_cache(path, item_size=item_size, token_len=token_len, token_dim=token_dim)
        return True
    except (FileNotFoundError, ValueError, RuntimeError):
        return False


def preprocessing_is_valid(manifest_path, poster_dir, expected_items):
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        return (
            int(manifest["num_items"]) == int(expected_items)
            and all((Path(poster_dir) / f"{item_id}.jpg").is_file() for item_id in range(1, expected_items + 1))
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def vlm_cache_is_valid(cache_path, meta_path, item_size, token_len, token_dim, expected_items, limit):
    if not cache_is_valid(cache_path, item_size, token_len, token_dim):
        return False


def compressed_cache_is_valid(cache_path, meta_path, item_size, token_len, token_dim, source_cache, config):
    if not cache_is_valid(cache_path, item_size, token_len, token_dim):
        return False
    try:
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        return (
            int(meta.get("input_dim", -1)) == int(config["input_dim"])
            and int(meta.get("bottleneck_dim", -1)) == int(config["bottleneck_dim"])
            and float(meta.get("lambda1", -1)) == float(config["lambda1"])
            and float(meta.get("lambda2", -1)) == float(config["lambda2"])
            and Path(meta.get("source_cache_path", "")).resolve() == Path(source_cache).resolve()
        )
    except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError):
        return False
    try:
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        return (
            meta.get("plan") == "avg_pool_start_end_noprompt"
            and meta.get("token_selection") == "special_tokens_0_last_plus_image_adaptive_avg_pool"
            and int(meta.get("embedded_count", -1)) == int(expected_items)
            and int(meta.get("error_count", -1)) == 0
            and meta.get("limit") == limit
        )
    except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError):
        return False


def parse_stages(value):
    if value == "all":
        return list(ALL_STAGES)
    stages = [stage.strip() for stage in value.split(",") if stage.strip()]
    invalid = sorted(set(stages) - set(PIPELINE_STAGES))
    if invalid:
        raise ValueError(f"Unknown stages: {invalid}")
    return stages


def main():
    parser = argparse.ArgumentParser(description="Run the standalone PixelRec pipeline.")
    parser.add_argument("--dataset", required=True, choices=("beauty", "games", "toys"))
    parser.add_argument("--images-dir")
    parser.add_argument("--qwen-model")
    parser.add_argument("--modelscope-id")
    parser.add_argument("--artifacts-dir")
    parser.add_argument("--stages", default="all", help="all or a comma-separated stage list")
    parser.add_argument("--force-stage", action="append", default=[], choices=PIPELINE_STAGES)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpu-device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--history", help="Required only for the recommend stage.")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    if args.images_dir:
        args.images_dir = str(Path(args.images_dir).expanduser().resolve())
    if args.qwen_model:
        args.qwen_model = str(Path(args.qwen_model).expanduser().resolve())
    if args.artifacts_dir:
        args.artifacts_dir = str(Path(args.artifacts_dir).expanduser().resolve())
    default_qwen_model = PROJECT_ROOT / "weights" / "vlm" / "Qwen3-VL-Embedding-8B"
    if not args.qwen_model and default_qwen_model.is_dir():
        args.qwen_model = str(default_qwen_model)

    config = load_dataset_config(args.dataset)
    item_size = int(config["num_items"]) + 1
    token_len = int(config["vlm"]["max_token_cache_len"])
    raw_dim = int(config["compression"]["input_dim"])
    compressed_dim = int(config["compression"]["bottleneck_dim"])
    root = Path(args.artifacts_dir) if args.artifacts_dir else PROJECT_ROOT / "artifacts" / args.dataset
    preprocess_dir = root / "preprocess"
    posters = preprocess_dir / "posters"
    raw_cache = root / "vlm" / "tokens_4096.pt"
    raw_meta = root / "vlm" / "tokens_4096.meta.json"
    compressed_cache = root / "compression" / "tokens_1024.pt"
    compressed_meta = root / "compression" / "tokens_1024.meta.json"
    compressor = root / "compression" / "compressor.pt"
    training_dir = root / "training"
    checkpoint = training_dir / "best.pt"
    stages = parse_stages(args.stages)
    force = set(args.force_stage)
    dependency_order = list(ALL_STAGES)
    for forced_stage in tuple(force):
        if forced_stage in dependency_order:
            force.update(dependency_order[dependency_order.index(forced_stage) :])
    expected_items = min(args.limit, int(config["num_items"])) if args.limit is not None else int(config["num_items"])

    if "preprocess" in stages:
        if not args.images_dir:
            parser.error("--images-dir is required for the preprocess stage")
        manifest = preprocess_dir / "preprocess_manifest.json"
        if preprocessing_is_valid(manifest, posters, expected_items) and "preprocess" not in force:
            print(f"Reusing preprocessing output: {manifest}")
        else:
            command = [sys.executable, "preprocess.py", "--dataset", args.dataset, "--images-dir", args.images_dir, "--output-dir", preprocess_dir]
            if "preprocess" in force:
                command.append("--force")
            if args.limit is not None:
                command.extend(["--limit", args.limit])
            run(command)

    if "vlm" in stages:
        if not args.qwen_model and not args.modelscope_id:
            parser.error("--qwen-model or --modelscope-id is required for the vlm stage")
        if vlm_cache_is_valid(raw_cache, raw_meta, item_size, token_len, raw_dim, expected_items, args.limit) and "vlm" not in force:
            print(f"Reusing VLM cache: {raw_cache}")
        else:
            raw_cache.parent.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable, "extract_vlm.py", "--data_dir", str(PROJECT_ROOT / "data" / args.dataset),
                "--data_name", args.dataset, "--image_dir", posters, "--meta_json", PROJECT_ROOT / "data" / args.dataset / "metadata.json",
                "--output_path", raw_cache, "--output_meta_json", raw_meta, "--instruction", "",
                "--max_token_cache_len", token_len, "--pooled_grid_h", config["vlm"]["pooled_grid_h"],
                "--pooled_grid_w", config["vlm"]["pooled_grid_w"], "--batch_size", config["vlm"]["batch_size"],
                "--device", args.gpu_device, "--dtype", config["vlm"]["dtype"],
                "--attn_implementation", config["vlm"]["attn_implementation"],
            ]
            if args.qwen_model:
                command.extend(["--model_name_or_path", args.qwen_model])
            else:
                command.extend(["--modelscope_model_id", args.modelscope_id])
            if args.limit is not None:
                command.extend(["--limit", args.limit])
            run(command)
            if not vlm_cache_is_valid(raw_cache, raw_meta, item_size, token_len, raw_dim, expected_items, args.limit):
                raise RuntimeError(f"VLM stage produced an invalid cache: {raw_cache}")

    if "compress" in stages:
        comp = config["compression"]
        if compressed_cache_is_valid(compressed_cache, compressed_meta, item_size, token_len, compressed_dim, raw_cache, comp) and "compress" not in force:
            print(f"Reusing compressed cache: {compressed_cache}")
        else:
            compressed_cache.parent.mkdir(parents=True, exist_ok=True)
            run([
                sys.executable, "compress.py", "--input_path", raw_cache, "--output_path", compressed_cache,
                "--checkpoint_path", compressor, "--output_meta_json", compressed_meta,
                "--input_dim", comp["input_dim"], "--bottleneck_dim", comp["bottleneck_dim"],
                "--lambda1", comp["lambda1"], "--lambda2", comp["lambda2"], "--batch_size", comp["batch_size"],
                "--epochs", comp["epochs"], "--train_tokens_per_epoch", comp["train_tokens_per_epoch"],
                "--lr", comp["lr"], "--weight_decay", comp["weight_decay"], "--seed", comp["seed"],
                "--device", args.gpu_device,
            ])
            if not compressed_cache_is_valid(compressed_cache, compressed_meta, item_size, token_len, compressed_dim, raw_cache, comp):
                raise RuntimeError(f"Compression stage produced an invalid cache: {compressed_cache}")

    if "train" in stages:
        if checkpoint.exists() and "train" not in force:
            print(f"Reusing checkpoint: {checkpoint}")
        else:
            command = [sys.executable, "train.py", "--dataset", args.dataset, "--cache", compressed_cache, "--output-dir", training_dir, "--device", args.device]
            if args.epochs is not None:
                command.extend(["--epochs", args.epochs])
            if args.batch_size is not None:
                command.extend(["--batch-size", args.batch_size])
            if args.num_workers is not None:
                command.extend(["--num-workers", args.num_workers])
            run(command)

    if "evaluate" in stages:
        run([
            sys.executable, "evaluate.py", "--dataset", args.dataset, "--cache", compressed_cache,
            "--checkpoint", checkpoint, "--output", root / "evaluation.json", "--device", args.device,
        ])

    if "recommend" in stages:
        if not args.history:
            parser.error("--history is required for the recommend stage")
        run([
            sys.executable, "recommend.py", "--dataset", args.dataset, "--cache", compressed_cache,
            "--checkpoint", checkpoint, "--history", args.history, "--top-k", args.top_k,
            "--output", root / "recommendations.json", "--device", args.device,
        ])

    manifest = {
        "format": "PixelRec-pipeline-v1",
        "dataset": args.dataset,
        "completed_stages": stages,
        "paths": {
            "posters": str(posters), "raw_cache": str(raw_cache), "compressed_cache": str(compressed_cache),
            "compressor": str(compressor), "checkpoint": str(checkpoint),
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "pipeline_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
