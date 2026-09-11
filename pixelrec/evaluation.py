import argparse
import json

from .data import make_dataloaders, rating_matrix
from .runtime import build_runtime, load_checkpoint, save_json
from .trainer import PixelRecTrainer


def evaluate(dataset, cache_path, checkpoint, output=None, device="auto", batch_size=None, num_workers=None):
    config, _, sequences, args, model, resolved_device = build_runtime(
        dataset,
        cache_path,
        device=device,
        overrides={"batch_size": batch_size, "num_workers": num_workers},
    )
    load_checkpoint(model, checkpoint, resolved_device)
    loaders = make_dataloaders(
        sequences, args.item_size, args.batch_size, args.max_seq_length, args.num_workers
    )
    seen = rating_matrix(sequences, args.item_size, "test")
    trainer = PixelRecTrainer(model, resolved_device, learning_rate=args.lr)
    metrics = trainer.evaluate(loaders["test"], seen)
    result = {"dataset": config["dataset"], "checkpoint": str(checkpoint), "metrics": metrics}
    if output:
        save_json(output, result)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a PixelRec checkpoint on the full test split.")
    parser.add_argument("--dataset", required=True, choices=("beauty", "games", "toys"))
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    result = evaluate(
        args.dataset, args.cache, args.checkpoint, args.output, args.device,
        args.batch_size, args.num_workers,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
