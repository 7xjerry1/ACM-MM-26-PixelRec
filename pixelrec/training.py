import argparse
import logging
from pathlib import Path

from .data import make_dataloaders, rating_matrix
from .runtime import build_runtime, load_checkpoint, save_json
from .trainer import PixelRecTrainer, save_checkpoint


def train(dataset, cache_path, output_dir, device="auto", epochs=None, batch_size=None, num_workers=None, resume=None):
    overrides = {
        "epochs": epochs,
        "batch_size": batch_size,
        "num_workers": num_workers,
    }
    config, _, sequences, args, model, resolved_device = build_runtime(
        dataset, cache_path, device=device, overrides=overrides
    )
    loaders = make_dataloaders(
        sequences, args.item_size, args.batch_size, args.max_seq_length, args.num_workers
    )
    valid_seen = rating_matrix(sequences, args.item_size, "valid")
    test_seen = rating_matrix(sequences, args.item_size, "test")
    trainer = PixelRecTrainer(
        model,
        resolved_device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
    )
    if resume:
        load_checkpoint(model, resume, resolved_device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_score = float("-inf")
    best_epoch = -1
    patience_count = 0
    history = []
    for epoch in range(args.epochs):
        loss = trainer.train_epoch(loaders["train"])
        metrics = trainer.evaluate(loaders["valid"], valid_seen)
        record = {"epoch": epoch, "train_loss": loss, "validation": metrics}
        history.append(record)
        logging.info("epoch=%s loss=%.6f validation=%s", epoch, loss, metrics)
        if metrics["N@20"] > best_score:
            best_score = metrics["N@20"]
            best_epoch = epoch
            patience_count = 0
            save_checkpoint(output_dir / "best.pt", model, config, epoch, metrics)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                break
    load_checkpoint(model, output_dir / "best.pt", resolved_device)
    test_metrics = trainer.evaluate(loaders["test"], test_seen)
    result = {
        "dataset": config["dataset"],
        "best_epoch": best_epoch,
        "validation_ndcg20": best_score,
        "test": test_metrics,
        "history": history,
    }
    save_json(output_dir / "train_results.json", result)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Train PixelRec.")
    parser.add_argument("--dataset", required=True, choices=("beauty", "games", "toys"))
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--resume")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    result = train(
        args.dataset, args.cache, args.output_dir, args.device, args.epochs,
        args.batch_size, args.num_workers, args.resume,
    )
    print(result["test"])


if __name__ == "__main__":
    main()
