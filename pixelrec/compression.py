#!/usr/bin/env python3
import argparse
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("pixelrec.rcc")


@dataclass
class TokenCache:
    tokens: torch.Tensor
    mask: torch.Tensor
    meta: Dict


@dataclass
class TokenSplit:
    train_pairs: torch.Tensor
    val_pairs: torch.Tensor
    val_item_ids: torch.Tensor
    num_items: int


class LinearTokenCompressor(nn.Module):
    def __init__(self, input_dim: int, bottleneck_dim: int):
        super().__init__()
        self.enc = nn.Linear(input_dim, bottleneck_dim, bias=False)
        self.dec = nn.Linear(bottleneck_dim, input_dim, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.enc(x)
        recon = self.dec(z)
        return z, recon


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a small linear compressor for Qwen3-VL token cache and export compressed tokens."
    )
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--output_meta_json", type=str, default=None)
    parser.add_argument("--input_dim", type=int, default=4096)
    parser.add_argument("--bottleneck_dim", type=int, default=1024)
    parser.add_argument("--lambda1", type=float, default=1.0)
    parser.add_argument("--lambda2", type=float, default=0.1)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2048,
        help="Number of randomly sampled tokens per optimization step.",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--train_tokens_per_epoch", type=int, default=524288)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--save_dtype",
        type=str,
        default="source",
        choices=["source", "float32", "float16", "bfloat16"],
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--limit_items",
        type=int,
        default=None,
        help="Only use the first N item rows for smoke/debug runs.",
    )
    parser.add_argument("--val_token_count", type=int, default=8192)
    parser.add_argument("--val_item_count", type=int, default=512)
    parser.add_argument("--val_interval", type=int, default=1000)
    parser.add_argument("--item_eval_batch_size", type=int, default=64)
    parser.add_argument("--val_item_cosine_min_delta", type=float, default=0.001)
    parser.add_argument("--val_item_cosine_patience", type=int, default=3)
    parser.add_argument("--val_recon_patience", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=20)
    return parser.parse_args()


def ensure_parent(path: Optional[str]):
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def resolve_paths(args):
    if args.output_path is None:
        input_dir = os.path.dirname(args.input_path)
        args.output_path = os.path.join(input_dir, "Qwen3_VL_token_cache_1024.pt")
    if args.checkpoint_path is None:
        root, _ = os.path.splitext(args.output_path)
        args.checkpoint_path = f"{root}_compressor.pt"
    if args.output_meta_json is None:
        root, _ = os.path.splitext(args.output_path)
        args.output_meta_json = f"{root}.meta.json"
    ensure_parent(args.output_path)
    ensure_parent(args.checkpoint_path)
    ensure_parent(args.output_meta_json)
    return args


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_token_cache(input_path: str) -> TokenCache:
    payload = torch.load(input_path, map_location="cpu")
    if isinstance(payload, dict):
        tokens = payload.get("tokens")
        mask = payload.get("mask")
    elif isinstance(payload, (list, tuple)) and len(payload) == 2:
        tokens, mask = payload
    else:
        raise ValueError(
            "vlm token cache must be a dict with `tokens`/`mask` or a 2-tuple `(tokens, mask)`."
        )
    if not isinstance(tokens, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise ValueError("vlm token cache must contain tensor `tokens` and `mask`.")
    meta_path = os.path.splitext(input_path)[0] + ".meta.json"
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    return TokenCache(tokens=tokens.contiguous(), mask=mask.bool().contiguous(), meta=meta)


def resolve_dtype(dtype_name: str, source_dtype: torch.dtype) -> torch.dtype:
    if dtype_name == "source":
        return source_dtype
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype_name]


def dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    return str(dtype).replace("torch.", "")


def resolve_num_items(cache: TokenCache, limit_items: Optional[int]) -> int:
    if limit_items is None:
        return int(cache.tokens.size(0))
    return min(int(cache.tokens.size(0)), int(limit_items) + 1)


def build_token_split(args, cache: TokenCache) -> Tuple[TokenSplit, Dict]:
    num_items = resolve_num_items(cache, args.limit_items)
    mask = cache.mask[:num_items]
    if num_items <= 1:
        raise ValueError("Need at least one non-padding item in the token cache.")

    valid_pairs = torch.nonzero(mask[1:], as_tuple=False)
    if valid_pairs.size(0) < 2:
        raise ValueError("Need at least two valid tokens to build train/validation splits.")
    valid_pairs = valid_pairs.to(dtype=torch.long)
    valid_pairs[:, 0] += 1

    valid_item_ids = torch.nonzero(mask.any(dim=1), as_tuple=False).squeeze(-1)
    valid_item_ids = valid_item_ids[valid_item_ids.gt(0)]
    if valid_item_ids.numel() == 0:
        raise ValueError("No valid items found in the token cache.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)

    pair_perm = torch.randperm(valid_pairs.size(0), generator=generator)
    val_token_count = min(int(args.val_token_count), int(valid_pairs.size(0) - 1))
    if val_token_count <= 0:
        raise ValueError("Validation token count must be positive and smaller than the number of valid tokens.")
    val_pairs = valid_pairs[pair_perm[:val_token_count]].contiguous()
    train_pairs = valid_pairs[pair_perm[val_token_count:]].contiguous()
    if train_pairs.size(0) == 0:
        raise ValueError("Training split has no valid tokens after reserving the validation token batch.")

    item_perm = torch.randperm(valid_item_ids.numel(), generator=generator)
    val_item_count = min(int(args.val_item_count), int(valid_item_ids.numel()))
    val_item_ids = valid_item_ids[item_perm[:val_item_count]].contiguous()

    split = TokenSplit(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        val_item_ids=val_item_ids,
        num_items=num_items,
    )
    split_meta = {
        "num_items": num_items,
        "num_valid_tokens": int(valid_pairs.size(0)),
        "num_train_tokens": int(train_pairs.size(0)),
        "num_val_tokens": int(val_pairs.size(0)),
        "num_valid_items": int(valid_item_ids.numel()),
        "num_val_items": int(val_item_ids.numel()),
    }
    return split, split_meta


def clone_model_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def gather_token_pairs(cache: TokenCache, token_pairs: torch.Tensor, device: torch.device) -> torch.Tensor:
    item_ids = token_pairs[:, 0]
    token_ids = token_pairs[:, 1]
    batch_tokens = cache.tokens[item_ids, token_ids]
    return batch_tokens.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")


def compute_batch_loss(
    model: LinearTokenCompressor,
    batch_tokens: torch.Tensor,
    lambda1: float,
    lambda2: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    _, recon = model(batch_tokens)
    smooth_l1 = F.smooth_l1_loss(recon, batch_tokens)
    cosine_penalty = 1.0 - F.cosine_similarity(recon, batch_tokens, dim=-1).mean()
    total = lambda1 * smooth_l1 + lambda2 * cosine_penalty
    stats = {
        "smooth_l1": float(smooth_l1.detach().item()),
        "cosine_penalty": float(cosine_penalty.detach().item()),
        "total": float(total.detach().item()),
    }
    return total, stats


def masked_item_mean(token_states: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    token_mask = token_mask.unsqueeze(-1).to(dtype=token_states.dtype)
    denom = token_mask.sum(dim=1).clamp_min(1.0)
    return (token_states * token_mask).sum(dim=1) / denom


@torch.no_grad()
def evaluate_model(
    args,
    cache: TokenCache,
    split: TokenSplit,
    model: LinearTokenCompressor,
    device: torch.device,
    global_step: int,
    epoch: int,
) -> Dict[str, float]:
    model.eval()

    val_tokens = gather_token_pairs(cache, split.val_pairs, device)
    _, val_recon = model(val_tokens)
    val_recon_loss = float(F.smooth_l1_loss(val_recon, val_tokens).item())

    item_cosine_sum = 0.0
    item_count = 0
    for start in range(0, split.val_item_ids.numel(), args.item_eval_batch_size):
        end = min(split.val_item_ids.numel(), start + args.item_eval_batch_size)
        item_ids = split.val_item_ids[start:end]
        batch_tokens = cache.tokens[item_ids].to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
        batch_mask = cache.mask[item_ids].to(device=device, non_blocking=device.type == "cuda")
        _, batch_recon = model(batch_tokens)
        item_orig = masked_item_mean(batch_tokens, batch_mask)
        item_recon = masked_item_mean(batch_recon, batch_mask)
        cosine = F.cosine_similarity(item_recon, item_orig, dim=-1)
        item_cosine_sum += float(cosine.sum().item())
        item_count += int(cosine.numel())

    val_item_cosine = item_cosine_sum / max(item_count, 1)
    metrics = {
        "global_step": int(global_step),
        "epoch": int(epoch),
        "val_recon_loss": val_recon_loss,
        "val_item_cosine": val_item_cosine,
    }
    LOGGER.info(
        "eval step=%s epoch=%s val_recon_loss=%.6f val_item_cosine=%.6f",
        metrics["global_step"],
        metrics["epoch"],
        metrics["val_recon_loss"],
        metrics["val_item_cosine"],
    )
    return metrics


def save_checkpoint(
    path: str,
    model_state_dict: Dict[str, torch.Tensor],
    args,
    split_meta: Dict,
    best_eval: Dict,
    eval_history: list,
    epoch_history: list,
):
    checkpoint = {
        "model_state_dict": model_state_dict,
        "config": {
            "input_dim": args.input_dim,
            "bottleneck_dim": args.bottleneck_dim,
            "lambda1": args.lambda1,
            "lambda2": args.lambda2,
            "epochs": args.epochs,
            "train_tokens_per_epoch": args.train_tokens_per_epoch,
            "batch_size": args.batch_size,
            "val_token_count": args.val_token_count,
            "val_item_count": args.val_item_count,
            "val_interval": args.val_interval,
            "item_eval_batch_size": args.item_eval_batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        },
        "split": split_meta,
        "best_eval": best_eval,
        "eval_history": eval_history,
        "epoch_history": epoch_history,
    }
    torch.save(checkpoint, path)


def run_validation(
    args,
    cache: TokenCache,
    split: TokenSplit,
    model: LinearTokenCompressor,
    device: torch.device,
    global_step: int,
    epoch: int,
    eval_history: list,
    epoch_history: list,
    split_meta: Dict,
    best_state_dict: Optional[Dict[str, torch.Tensor]],
    best_eval: Optional[Dict[str, float]],
    best_val_item_cosine: float,
    best_val_recon_loss: float,
    small_item_gain_count: int,
    recon_no_decrease_count: int,
) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[Dict[str, float]], float, float, int, int, bool]:
    metrics = evaluate_model(
        args=args,
        cache=cache,
        split=split,
        model=model,
        device=device,
        global_step=global_step,
        epoch=epoch,
    )
    eval_history.append(metrics)

    if best_eval is None:
        item_gain = float("inf")
        recon_improved = True
    else:
        item_gain = metrics["val_item_cosine"] - best_val_item_cosine
        recon_improved = metrics["val_recon_loss"] < best_val_recon_loss

    if best_eval is None or item_gain >= args.val_item_cosine_min_delta:
        small_item_gain_count = 0
    else:
        small_item_gain_count += 1

    if best_eval is None or recon_improved:
        recon_no_decrease_count = 0
    else:
        recon_no_decrease_count += 1

    if metrics["val_item_cosine"] > best_val_item_cosine:
        best_val_item_cosine = metrics["val_item_cosine"]
    if metrics["val_recon_loss"] < best_val_recon_loss:
        best_val_recon_loss = metrics["val_recon_loss"]

    is_new_best = (
        best_eval is None
        or metrics["val_item_cosine"] > best_eval["val_item_cosine"]
        or (
            metrics["val_item_cosine"] == best_eval["val_item_cosine"]
            and metrics["val_recon_loss"] < best_eval["val_recon_loss"]
        )
    )
    if is_new_best:
        best_state_dict = clone_model_state_dict(model)
        best_eval = dict(metrics)
        best_eval["best_reason"] = "max_val_item_cosine_then_min_val_recon_loss"
        save_checkpoint(
            path=args.checkpoint_path,
            model_state_dict=best_state_dict,
            args=args,
            split_meta=split_meta,
            best_eval=best_eval,
            eval_history=eval_history,
            epoch_history=epoch_history,
        )

    should_stop = (
        small_item_gain_count >= args.val_item_cosine_patience
        or recon_no_decrease_count >= args.val_recon_patience
    )
    if should_stop:
        LOGGER.info(
            "early stop triggered at step=%s epoch=%s small_item_gain_count=%s recon_no_decrease_count=%s",
            global_step,
            epoch,
            small_item_gain_count,
            recon_no_decrease_count,
        )

    model.train()
    return (
        best_state_dict,
        best_eval,
        best_val_item_cosine,
        best_val_recon_loss,
        small_item_gain_count,
        recon_no_decrease_count,
        should_stop,
    )


def train_compressor(args, cache: TokenCache) -> Tuple[LinearTokenCompressor, Dict, Dict]:
    split, split_meta = build_token_split(args, cache)
    source_dtype = cache.tokens.dtype
    device = torch.device(args.device)
    model = LinearTokenCompressor(args.input_dim, args.bottleneck_dim).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_generator = torch.Generator(device="cpu")
    train_generator.manual_seed(args.seed + 1)

    steps_per_epoch = max(1, math.ceil(args.train_tokens_per_epoch / args.batch_size))
    LOGGER.info(
        "training compressor input_dim=%s bottleneck_dim=%s source_dtype=%s items=%s train_tokens=%s val_tokens=%s val_items=%s batch_size=%s steps_per_epoch=%s epochs=%s device=%s",
        args.input_dim,
        args.bottleneck_dim,
        dtype_name(source_dtype),
        split_meta["num_items"] - 1,
        split_meta["num_train_tokens"],
        split_meta["num_val_tokens"],
        split_meta["num_val_items"],
        args.batch_size,
        steps_per_epoch,
        args.epochs,
        device,
    )

    eval_history = []
    epoch_history = []
    best_state_dict = None
    best_eval = None
    best_val_item_cosine = float("-inf")
    best_val_recon_loss = float("inf")
    small_item_gain_count = 0
    recon_no_decrease_count = 0
    global_step = 0
    last_eval_step = 0
    early_stop_reason = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_total = 0.0
        running_recon = 0.0
        running_cosine_penalty = 0.0
        running_tokens = 0

        progress = tqdm.tqdm(range(steps_per_epoch), desc=f"Train epoch {epoch}")
        for epoch_step in progress:
            sample_idx = torch.randint(
                low=0,
                high=split.train_pairs.size(0),
                size=(args.batch_size,),
                generator=train_generator,
            )
            batch_pairs = split.train_pairs[sample_idx]
            batch_tokens = gather_token_pairs(cache, batch_pairs, device)

            optimizer.zero_grad(set_to_none=True)
            loss, stats = compute_batch_loss(
                model=model,
                batch_tokens=batch_tokens,
                lambda1=args.lambda1,
                lambda2=args.lambda2,
            )
            loss.backward()
            optimizer.step()

            batch_token_count = int(batch_tokens.size(0))
            running_total += stats["total"] * batch_token_count
            running_recon += stats["smooth_l1"] * batch_token_count
            running_cosine_penalty += stats["cosine_penalty"] * batch_token_count
            running_tokens += batch_token_count
            global_step += 1

            if global_step % args.log_interval == 0 or epoch_step == 0:
                progress.set_postfix(
                    step=global_step,
                    loss=f"{stats['total']:.4f}",
                    recon=f"{stats['smooth_l1']:.4f}",
                    cosine_pen=f"{stats['cosine_penalty']:.4f}",
                )

            if global_step % args.val_interval == 0:
                (
                    best_state_dict,
                    best_eval,
                    best_val_item_cosine,
                    best_val_recon_loss,
                    small_item_gain_count,
                    recon_no_decrease_count,
                    should_stop,
                ) = run_validation(
                    args=args,
                    cache=cache,
                    split=split,
                    model=model,
                    device=device,
                    global_step=global_step,
                    epoch=epoch,
                    eval_history=eval_history,
                    epoch_history=epoch_history,
                    split_meta=split_meta,
                    best_state_dict=best_state_dict,
                    best_eval=best_eval,
                    best_val_item_cosine=best_val_item_cosine,
                    best_val_recon_loss=best_val_recon_loss,
                    small_item_gain_count=small_item_gain_count,
                    recon_no_decrease_count=recon_no_decrease_count,
                )
                last_eval_step = global_step
                if should_stop:
                    if small_item_gain_count >= args.val_item_cosine_patience:
                        early_stop_reason = "val_item_cosine_small_gain"
                    else:
                        early_stop_reason = "val_recon_loss_no_decrease"
                    break

        epoch_stats = {
            "epoch": epoch,
            "global_step": global_step,
            "mean_total_loss": running_total / max(running_tokens, 1),
            "mean_smooth_l1": running_recon / max(running_tokens, 1),
            "mean_cosine_penalty": running_cosine_penalty / max(running_tokens, 1),
            "num_tokens": running_tokens,
            "steps": steps_per_epoch,
        }
        epoch_history.append(epoch_stats)
        LOGGER.info(
            "epoch=%s global_step=%s mean_total_loss=%.6f mean_smooth_l1=%.6f mean_cosine_penalty=%.6f num_tokens=%s",
            epoch_stats["epoch"],
            epoch_stats["global_step"],
            epoch_stats["mean_total_loss"],
            epoch_stats["mean_smooth_l1"],
            epoch_stats["mean_cosine_penalty"],
            epoch_stats["num_tokens"],
        )

        if early_stop_reason is not None:
            break

    if last_eval_step != global_step or best_eval is None:
        (
            best_state_dict,
            best_eval,
            best_val_item_cosine,
            best_val_recon_loss,
            small_item_gain_count,
            recon_no_decrease_count,
            _,
        ) = run_validation(
            args=args,
            cache=cache,
            split=split,
            model=model,
            device=device,
            global_step=global_step,
            epoch=epoch_history[-1]["epoch"] if epoch_history else 0,
            eval_history=eval_history,
            epoch_history=epoch_history,
            split_meta=split_meta,
            best_state_dict=best_state_dict,
            best_eval=best_eval,
            best_val_item_cosine=best_val_item_cosine,
            best_val_recon_loss=best_val_recon_loss,
            small_item_gain_count=small_item_gain_count,
            recon_no_decrease_count=recon_no_decrease_count,
        )

    if best_state_dict is None or best_eval is None:
        raise RuntimeError("Training finished without a valid checkpoint.")

    model.load_state_dict(best_state_dict)
    training_state = {
        "input_dim": args.input_dim,
        "bottleneck_dim": args.bottleneck_dim,
        "lambda1": args.lambda1,
        "lambda2": args.lambda2,
        "epochs": args.epochs,
        "train_tokens_per_epoch": args.train_tokens_per_epoch,
        "batch_size": args.batch_size,
        "val_token_count": args.val_token_count,
        "val_item_count": args.val_item_count,
        "val_interval": args.val_interval,
        "item_eval_batch_size": args.item_eval_batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "source_dtype": dtype_name(source_dtype),
        "split": split_meta,
        "best_eval": best_eval,
        "eval_history": eval_history,
        "epoch_history": epoch_history,
        "early_stop_reason": early_stop_reason or "max_epochs",
        "final_global_step": global_step,
    }
    return model, training_state, split_meta


def iterate_item_batches(num_items: int, batch_size: int):
    for start in range(1, num_items, batch_size):
        yield start, min(num_items, start + batch_size)


@torch.no_grad()
def export_compressed_cache(
    args,
    cache: TokenCache,
    model: LinearTokenCompressor,
    split_meta: Dict,
) -> Tuple[Dict, Dict]:
    num_items = resolve_num_items(cache, args.limit_items)
    source_dtype = cache.tokens.dtype
    save_dtype = resolve_dtype(args.save_dtype, source_dtype=source_dtype)
    device = next(model.parameters()).device
    model.eval()

    compressed_tokens = torch.zeros(
        num_items,
        cache.tokens.size(1),
        args.bottleneck_dim,
        dtype=save_dtype,
    )
    compressed_mask = cache.mask[:num_items].clone()

    total_batches = math.ceil(max(num_items - 1, 0) / args.item_eval_batch_size)
    progress = tqdm.tqdm(
        iterate_item_batches(num_items=num_items, batch_size=args.item_eval_batch_size),
        total=total_batches,
        desc="Export compressed cache",
    )
    for start, end in progress:
        item_ids = torch.arange(start, end, dtype=torch.long)
        batch_tokens = cache.tokens[item_ids].to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
        batch_mask = cache.mask[item_ids]
        z = model.enc(batch_tokens).cpu().to(dtype=save_dtype)
        z = z * batch_mask.unsqueeze(-1).to(dtype=z.dtype)
        compressed_tokens[item_ids] = z

    payload = {
        "tokens": compressed_tokens,
        "mask": compressed_mask,
    }
    meta = {
        "pipeline": "Qwen3-VL token cache linear compression",
        "source_cache_path": args.input_path,
        "output_path": args.output_path,
        "checkpoint_path": args.checkpoint_path,
        "save_dtype": dtype_name(save_dtype),
        "source_dtype": dtype_name(source_dtype),
        "input_dim": args.input_dim,
        "bottleneck_dim": args.bottleneck_dim,
        "token_shape": list(compressed_tokens.shape),
        "mask_shape": list(compressed_mask.shape),
        "lambda1": args.lambda1,
        "lambda2": args.lambda2,
        "epochs": args.epochs,
        "train_tokens_per_epoch": args.train_tokens_per_epoch,
        "batch_size": args.batch_size,
        "val_token_count": args.val_token_count,
        "val_item_count": args.val_item_count,
        "val_interval": args.val_interval,
        "item_eval_batch_size": args.item_eval_batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "device": str(device),
        "seed": args.seed,
        "split": split_meta,
        "source_meta": cache.meta,
    }
    return payload, meta


def main():
    args = resolve_paths(parse_args())
    set_seed(args.seed)

    cache = load_token_cache(args.input_path)
    if cache.tokens.ndim != 3 or cache.mask.ndim != 2:
        raise ValueError("Expected token cache tensors shaped [num_items, max_tokens, dim] and [num_items, max_tokens].")
    if cache.tokens.size(2) != args.input_dim:
        raise ValueError(
            f"input token dim ({cache.tokens.size(2)}) != --input_dim ({args.input_dim})."
        )
    if cache.mask.size(0) != cache.tokens.size(0) or cache.mask.size(1) != cache.tokens.size(1):
        raise ValueError(
            f"mask shape {tuple(cache.mask.shape)} is incompatible with token shape {tuple(cache.tokens.shape)}."
        )

    model, train_state, split_meta = train_compressor(args, cache)
    payload, meta = export_compressed_cache(args, cache, model, split_meta=split_meta)
    meta["training"] = train_state

    checkpoint = {
        "model_state_dict": clone_model_state_dict(model),
        "config": {
            "input_dim": args.input_dim,
            "bottleneck_dim": args.bottleneck_dim,
            "lambda1": args.lambda1,
            "lambda2": args.lambda2,
            "epochs": args.epochs,
            "train_tokens_per_epoch": args.train_tokens_per_epoch,
            "batch_size": args.batch_size,
            "val_token_count": args.val_token_count,
            "val_item_count": args.val_item_count,
            "val_interval": args.val_interval,
            "item_eval_batch_size": args.item_eval_batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        },
        "split": split_meta,
        "training": train_state,
    }

    torch.save(payload, args.output_path)
    torch.save(checkpoint, args.checkpoint_path)
    with open(args.output_meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    LOGGER.info(
        "saved compressed token cache: %s shape=%s dtype=%s",
        args.output_path,
        tuple(payload["tokens"].shape),
        payload["tokens"].dtype,
    )
    LOGGER.info("saved compressor checkpoint: %s", args.checkpoint_path)
    LOGGER.info("saved metadata: %s", args.output_meta_json)


if __name__ == "__main__":
    main()
