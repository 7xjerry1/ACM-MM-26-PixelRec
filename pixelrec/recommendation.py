import argparse
import json

import torch

from .data import load_catalog, normalize_history, pad_history
from .runtime import build_runtime, load_checkpoint, save_json


@torch.no_grad()
def score_topk(model, input_ids, item_history, by_id, top_k):
    model.eval()
    sequence_output = model.predict(input_ids)[:, -1, :]
    scores = torch.matmul(sequence_output, model.get_test_item_emb().transpose(0, 1))[0]
    scores[0] = -torch.inf
    scores[item_history] = -torch.inf
    k = min(int(top_k), int(torch.isfinite(scores).sum().item()))
    values, indices = torch.topk(scores, k=k)
    recommendations = []
    for score, item_id in zip(values.cpu().tolist(), indices.cpu().tolist()):
        item = by_id[item_id]
        recommendations.append(
            {"score": score, "item_id": item_id, "asin": item["asin"], "title": item["title"]}
        )
    return recommendations


@torch.no_grad()
def recommend(dataset, cache_path, checkpoint, history, top_k=20, output=None, device="auto"):
    config, paths, _, args, model, resolved_device = build_runtime(dataset, cache_path, device=device)
    by_id, by_asin = load_catalog(paths["catalog"])
    item_history = normalize_history(history, by_asin, args.item_size, args.max_seq_length)
    input_ids = torch.tensor([pad_history(item_history, args.max_seq_length)], device=resolved_device)
    load_checkpoint(model, checkpoint, resolved_device)
    recommendations = score_topk(model, input_ids, item_history, by_id, top_k)
    k = len(recommendations)
    result = {
        "dataset": config["dataset"],
        "history_item_ids": item_history,
        "top_k": k,
        "recommendations": recommendations,
    }
    if output:
        save_json(output, result)
    return result


def parse_history(value):
    if value.startswith("["):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError("JSON history must be a list.")
        return parsed
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Top-K PixelRec recommendations from item IDs or ASINs.")
    parser.add_argument("--dataset", required=True, choices=("beauty", "games", "toys"))
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--history", required=True, help="Comma-separated values or a JSON list of item IDs/ASINs.")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    result = recommend(
        args.dataset, args.cache, args.checkpoint, parse_history(args.history),
        args.top_k, args.output, args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
