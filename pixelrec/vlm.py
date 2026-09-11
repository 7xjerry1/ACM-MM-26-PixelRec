#!/usr/bin/env python3
import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import tqdm
from modelscope import snapshot_download
from PIL import Image
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLModel,
    Qwen3VLPreTrainedModel,
)
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("pixelrec.vlm")

MAX_LENGTH = 8192
IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR
DEFAULT_INSTRUCTION = ""
SPECIAL_TOKEN_POSITIONS = [0, -1]
SPECIAL_TOKEN_PREFIX_POSITIONS = [0]
SPECIAL_TOKEN_SUFFIX_COUNT = 1


@dataclass
class Qwen3VLForTokenCacheOutput:
    last_hidden_state: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None


class Qwen3VLForTokenCache(Qwen3VLPreTrainedModel):
    _checkpoint_conversion_mapping = {}
    accepts_loss_kwargs = False
    config: Qwen3VLConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Qwen3VLForTokenCacheOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            **kwargs,
        )
        return Qwen3VLForTokenCacheOutput(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=attention_mask,
        )


class PixelRecVLMExtractor:
    def __init__(
        self,
        model_name_or_path: str,
        device: str,
        torch_dtype: Optional[torch.dtype],
        max_length: int = MAX_LENGTH,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        default_instruction: str = DEFAULT_INSTRUCTION,
        attn_implementation: Optional[str] = "sdpa",
    ):
        self.device = torch.device(device)
        self.max_length = max_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.default_instruction = default_instruction

        kwargs = {"trust_remote_code": True}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation

        self.model = Qwen3VLForTokenCache.from_pretrained(model_name_or_path, **kwargs)
        self.model.to(self.device)
        self.model.eval()
        self.processor = Qwen3VLProcessor.from_pretrained(model_name_or_path, padding_side="right")
        self.image_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)
        self.spatial_merge_size = int(self.model.config.vision_config.spatial_merge_size)

    def format_model_input(self, image: str, instruction: Optional[str]) -> List[Dict[str, Any]]:
        del instruction
        content = [
            {
                "type": "image",
                "image": image if image.startswith(("http://", "https://")) else f"file://{image}",
            }
        ]
        return [{"role": "user", "content": content}]

    def _load_images(self, image_paths: List[str]) -> List[Image.Image]:
        loaded = []
        for path in image_paths:
            with Image.open(path) as image:
                loaded.append(image.convert("RGB"))
        return loaded

    def preprocess(self, conversations: List[List[Dict[str, Any]]], image_paths: List[str]) -> Dict[str, torch.Tensor]:
        text = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=False,
        )
        images = self._load_images(image_paths)
        inputs = self.processor(
            text=text,
            images=images,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    @staticmethod
    def pool_image_tokens_with_special_tokens(
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        image_token_id: int,
        spatial_merge_size: int,
        pooled_grid_h: int,
        pooled_grid_w: int,
        max_tokens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, int]]]:
        pooled_token_count = pooled_grid_h * pooled_grid_w
        expected_total = len(SPECIAL_TOKEN_POSITIONS) + pooled_token_count
        if expected_total != max_tokens:
            raise ValueError(
                f"special-token avg-pool expects max_tokens == len(SPECIAL_TOKEN_POSITIONS) + pooled_grid_h * pooled_grid_w, got "
                f"{max_tokens} vs {len(SPECIAL_TOKEN_POSITIONS)} + {pooled_grid_h}*{pooled_grid_w} = {expected_total}."
            )

        batch_tokens = []
        batch_mask = []
        batch_grid_meta: List[Dict[str, int]] = []
        for hidden, mask, ids, grid_thw in zip(last_hidden_state, attention_mask, input_ids, image_grid_thw):
            valid_mask = mask.bool()
            valid_hidden = hidden[valid_mask]
            valid_ids = ids[valid_mask]
            valid_len = int(valid_hidden.size(0))
            if valid_len <= SPECIAL_TOKEN_PREFIX_POSITIONS[-1]:
                raise RuntimeError(
                    f"Sequence too short for special token positions {SPECIAL_TOKEN_POSITIONS}: valid_len={valid_len}."
                )

            last_position = valid_len - 1
            prefix_hidden = valid_hidden[SPECIAL_TOKEN_PREFIX_POSITIONS]
            suffix_hidden = valid_hidden[last_position:last_position + 1]
            special_ids = torch.cat(
                [
                    valid_ids[SPECIAL_TOKEN_PREFIX_POSITIONS],
                    valid_ids[last_position:last_position + 1],
                ],
                dim=0,
            )
            if (special_ids == image_token_id).any():
                raise RuntimeError(
                    "Encountered an image token at one of the preserved special-token positions "
                    f"{SPECIAL_TOKEN_POSITIONS}; token_ids={special_ids.tolist()}."
                )

            image_hidden = valid_hidden[valid_ids == image_token_id]
            if image_hidden.ndim != 2 or image_hidden.size(0) == 0:
                raise RuntimeError("No image tokens found in Qwen3-VL sequence for the current sample.")

            grid_t, grid_h, grid_w = [int(value.item()) for value in grid_thw]
            merged_h = grid_h // spatial_merge_size
            merged_w = grid_w // spatial_merge_size
            expected_image_tokens = grid_t * merged_h * merged_w
            if image_hidden.size(0) != expected_image_tokens:
                raise RuntimeError(
                    "Image token count does not match expected merged visual token count: "
                    f"found={image_hidden.size(0)} expected={expected_image_tokens} "
                    f"(grid_t={grid_t}, grid_h={grid_h}, grid_w={grid_w}, spatial_merge_size={spatial_merge_size})."
                )
            if grid_t != 1:
                raise RuntimeError(
                    f"Expected still-image inputs with grid_t=1, but got grid_t={grid_t}."
                )

            feature_map = image_hidden.view(merged_h, merged_w, hidden.size(-1))
            feature_map = feature_map.permute(2, 0, 1).unsqueeze(0)
            pooled = F.adaptive_avg_pool2d(feature_map, output_size=(pooled_grid_h, pooled_grid_w))
            pooled = pooled.squeeze(0).permute(1, 2, 0).reshape(pooled_token_count, hidden.size(-1))

            selected = hidden.new_zeros((max_tokens, hidden.size(-1)))
            selected_mask = torch.zeros(max_tokens, dtype=torch.bool, device=hidden.device)
            selected[0:len(SPECIAL_TOKEN_PREFIX_POSITIONS)] = prefix_hidden
            selected[len(SPECIAL_TOKEN_PREFIX_POSITIONS):len(SPECIAL_TOKEN_PREFIX_POSITIONS) + pooled_token_count] = pooled
            selected[len(SPECIAL_TOKEN_PREFIX_POSITIONS) + pooled_token_count:] = suffix_hidden
            selected_mask[:] = True

            batch_tokens.append(selected.cpu())
            batch_mask.append(selected_mask.cpu())
            batch_grid_meta.append(
                {
                    "grid_t": grid_t,
                    "grid_h": grid_h,
                    "grid_w": grid_w,
                    "merged_h": merged_h,
                    "merged_w": merged_w,
                    "image_token_count": int(image_hidden.size(0)),
                    "sequence_length": valid_len,
                    "prefix_special_positions": SPECIAL_TOKEN_PREFIX_POSITIONS,
                    "suffix_special_position": last_position,
                }
            )
        return torch.stack(batch_tokens, dim=0), torch.stack(batch_mask, dim=0), batch_grid_meta

    @torch.no_grad()
    def process(
        self,
        image_paths: List[str],
        instruction: Optional[str],
        max_tokens: int,
        pooled_grid_h: int,
        pooled_grid_w: int,
    ):
        conversations = [self.format_model_input(image=path, instruction=instruction) for path in image_paths]
        inputs = self.preprocess(conversations, image_paths)
        outputs = self.model(**inputs)
        if outputs.last_hidden_state is None:
            raise RuntimeError("Qwen3-VL model did not return last_hidden_state.")
        return self.pool_image_tokens_with_special_tokens(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=outputs.attention_mask,
            input_ids=inputs["input_ids"],
            image_grid_thw=inputs["image_grid_thw"],
            image_token_id=self.image_token_id,
            spatial_merge_size=self.spatial_merge_size,
            pooled_grid_h=pooled_grid_h,
            pooled_grid_w=pooled_grid_w,
            max_tokens=max_tokens,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract Qwen3-VL token cache with preserved special tokens start/last plus image adaptive avg pooling and no text prompt."
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--data_name", type=str, required=True, choices=["beauty", "games", "toys"])
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--meta_json", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--output_meta_json", type=str, default=None)
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--modelscope_model_id", type=str, default=None)
    parser.add_argument("--modelscope_cache_dir", type=str, default=None)
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--min_pixels", type=int, default=MIN_PIXELS)
    parser.add_argument("--max_pixels", type=int, default=MAX_PIXELS)
    parser.add_argument("--max_token_cache_len", type=int, default=34)
    parser.add_argument("--pooled_grid_h", type=int, default=8)
    parser.add_argument("--pooled_grid_w", type=int, default=4)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    expected_total = len(SPECIAL_TOKEN_POSITIONS) + args.pooled_grid_h * args.pooled_grid_w
    if args.max_token_cache_len != expected_total:
        raise ValueError(
            f"special-token avg-pool requires max_token_cache_len == len(SPECIAL_TOKEN_POSITIONS) + pooled_grid_h * pooled_grid_w, got "
            f"{args.max_token_cache_len} vs {len(SPECIAL_TOKEN_POSITIONS)} + {args.pooled_grid_h}*{args.pooled_grid_w} = {expected_total}."
        )
    return args


def resolve_paths(args):
    image_dir = args.image_dir or os.path.join(args.data_dir, "images_title_concat")
    meta_json = args.meta_json or os.path.join(args.data_dir, f"{args.data_name}_meta.json")
    output_path = args.output_path or os.path.join(args.data_dir, "Qwen3_VL_token_cache_avg_pool_start_end_noprompt.pt")
    output_meta_json = args.output_meta_json or os.path.splitext(output_path)[0] + ".meta.json"
    return image_dir, meta_json, output_path, output_meta_json


def ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_num_items(meta_json: str) -> int:
    with open(meta_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return int(meta["num_items"])


def torch_dtype_from_arg(dtype: str) -> Optional[torch.dtype]:
    if dtype == "auto":
        return None
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def disk_dtype_from_arg(dtype: str) -> torch.dtype:
    mapped = torch_dtype_from_arg(dtype)
    return torch.float32 if mapped is None else mapped


def batched(seq: List[int], batch_size: int):
    for start in range(0, len(seq), batch_size):
        yield seq[start:start + batch_size]


def resolve_model_path(args):
    if args.model_name_or_path:
        return args.model_name_or_path
    if args.modelscope_model_id:
        model_dir = snapshot_download(model_id=args.modelscope_model_id, cache_dir=args.modelscope_cache_dir)
        LOGGER.info("Downloaded ModelScope model to %s", model_dir)
        return model_dir
    raise ValueError("Pass --model_name_or_path or --modelscope_model_id.")


@torch.no_grad()
def main():
    args = parse_args()
    image_dir, meta_json, output_path, output_meta_json = resolve_paths(args)

    missing = [path for path in [image_dir, meta_json] if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")
    ensure_parent(output_path)
    ensure_parent(output_meta_json)

    num_items = load_num_items(meta_json)
    item_ids = [item_id for item_id in range(1, num_items + 1) if os.path.exists(os.path.join(image_dir, f"{item_id}.jpg"))]
    if args.limit is not None:
        item_ids = item_ids[: args.limit]

    model_path = resolve_model_path(args)
    LOGGER.info(
        "num_items=%s images_found=%s image_dir=%s model=%s max_token_cache_len=%s pooled_grid=%sx%s special_positions=%s selection=avg_pool_start_end_noprompt",
        num_items,
        len(item_ids),
        image_dir,
        model_path,
        args.max_token_cache_len,
        args.pooled_grid_h,
        args.pooled_grid_w,
        SPECIAL_TOKEN_POSITIONS,
    )

    embedder = PixelRecVLMExtractor(
        model_name_or_path=model_path,
        device=args.device,
        torch_dtype=torch_dtype_from_arg(args.dtype),
        max_length=args.max_length,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        attn_implementation=args.attn_implementation,
    )
    disk_dtype = disk_dtype_from_arg(args.dtype)

    tokens = None
    mask = None
    embedded_count = 0
    error_items: Dict[int, str] = {}
    merged_grid_histogram: Dict[str, int] = {}

    for batch_ids in tqdm.tqdm(list(batched(item_ids, args.batch_size)), desc="Extracting Qwen3-VL token cache avg_pool_start_end_noprompt"):
        batch_paths = [os.path.join(image_dir, f"{item_id}.jpg") for item_id in batch_ids]
        try:
            batch_tokens, batch_mask, batch_grid_meta = embedder.process(
                batch_paths,
                instruction=args.instruction,
                max_tokens=args.max_token_cache_len,
                pooled_grid_h=args.pooled_grid_h,
                pooled_grid_w=args.pooled_grid_w,
            )
            batch_tokens = batch_tokens.to(dtype=disk_dtype)
            if tokens is None:
                tokens = torch.zeros(
                    num_items + 1,
                    args.max_token_cache_len,
                    batch_tokens.size(2),
                    dtype=batch_tokens.dtype,
                )
                mask = torch.zeros(num_items + 1, args.max_token_cache_len, dtype=torch.bool)
            tokens[batch_ids] = batch_tokens
            mask[batch_ids] = batch_mask
            for grid_meta in batch_grid_meta:
                key = f"{grid_meta['merged_h']}x{grid_meta['merged_w']}"
                merged_grid_histogram[key] = merged_grid_histogram.get(key, 0) + 1
            embedded_count += len(batch_ids)
        except Exception as exc:
            LOGGER.warning("Batch failed for ids=%s; retrying one by one. error=%s", batch_ids, exc)
            for item_id, path in zip(batch_ids, batch_paths):
                try:
                    single_tokens, single_mask, single_grid_meta = embedder.process(
                        [path],
                        instruction=args.instruction,
                        max_tokens=args.max_token_cache_len,
                        pooled_grid_h=args.pooled_grid_h,
                        pooled_grid_w=args.pooled_grid_w,
                    )
                    single_tokens = single_tokens.to(dtype=disk_dtype)
                    if tokens is None:
                        tokens = torch.zeros(
                            num_items + 1,
                            args.max_token_cache_len,
                            single_tokens.size(2),
                            dtype=single_tokens.dtype,
                        )
                        mask = torch.zeros(num_items + 1, args.max_token_cache_len, dtype=torch.bool)
                    tokens[item_id] = single_tokens[0]
                    mask[item_id] = single_mask[0]
                    for grid_meta in single_grid_meta:
                        key = f"{grid_meta['merged_h']}x{grid_meta['merged_w']}"
                        merged_grid_histogram[key] = merged_grid_histogram.get(key, 0) + 1
                    embedded_count += 1
                except Exception as inner_exc:
                    error_items[item_id] = f"{type(inner_exc).__name__}: {inner_exc}"

    if tokens is None or mask is None:
        raise RuntimeError("No token cache was generated. Check image_dir and model loading.")

    payload = {
        "tokens": tokens,
        "mask": mask,
    }
    torch.save(payload, output_path)
    meta = {
        "pipeline": "PixelRec posters -> Qwen3-VL first/last tokens + adaptive average pooled visual tokens",
        "plan": "avg_pool_start_end_noprompt",
        "data_dir": args.data_dir,
        "data_name": args.data_name,
        "image_dir": image_dir,
        "meta_json": meta_json,
        "output_path": output_path,
        "model_name_or_path": model_path,
        "modelscope_model_id": args.modelscope_model_id,
        "modelscope_cache_dir": args.modelscope_cache_dir,
        "instruction": args.instruction if args.instruction is not None else DEFAULT_INSTRUCTION,
        "no_text_prompt": True,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "max_token_cache_len": args.max_token_cache_len,
        "pooled_grid_h": args.pooled_grid_h,
        "pooled_grid_w": args.pooled_grid_w,
        "special_token_positions": SPECIAL_TOKEN_POSITIONS,
        "special_token_prefix_positions": SPECIAL_TOKEN_PREFIX_POSITIONS,
        "special_token_suffix_count": SPECIAL_TOKEN_SUFFIX_COUNT,
        "dtype": args.dtype,
        "device": args.device,
        "attn_implementation": args.attn_implementation,
        "limit": args.limit,
        "num_items": num_items,
        "images_found": len(item_ids),
        "embedded_count": embedded_count,
        "error_count": len(error_items),
        "token_selection": "special_tokens_0_last_plus_image_adaptive_avg_pool",
        "source_merged_grid_histogram": merged_grid_histogram,
        "token_shape": list(tokens.shape),
        "mask_shape": list(mask.shape),
        "hidden_dim": int(tokens.size(2)),
        "spatial_merge_size": embedder.spatial_merge_size,
        "image_token_id": int(embedder.image_token_id),
        "error_items": error_items,
    }
    with open(output_meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    LOGGER.info("Saved Qwen3-VL token cache avg_pool_start_end_noprompt: %s token_shape=%s", output_path, tuple(tokens.shape))
    LOGGER.info("Saved metadata: %s", output_meta_json)
    if error_items:
        raise RuntimeError(
            f"VLM extraction failed for {len(error_items)} items; inspect {output_meta_json}."
        )


if __name__ == "__main__":
    main()
