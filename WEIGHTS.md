# Downloaded Weights and Dataset Assets

This guide defines where externally hosted files belong and how to verify that they are complete and reproduce the reported PixelRec results.

## Download Links

Replace the placeholders below when the public files are available.

| Asset | Download |
|---|---|
| Qwen3-VL-Embedding-8B | [Official model page](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) or `GOOGLE_DRIVE_VLM_URL` |
| PixelRec checkpoints, RCC checkpoints, and compressed caches | `GOOGLE_DRIVE_PIXELREC_WEIGHTS_URL` |
| Beauty, Games, and Toys image archives | `GOOGLE_DRIVE_DATASETS_URL` |

The exact interaction sequences, item IDs, ASIN mappings, titles, configurations, and expected metrics are already included in this repository. The external dataset archive only needs to supply the large product-image directories.

## Expected Directory Layout

Extract downloaded files without changing their dataset names:

```text
PixelRec/
├── weights/
│   ├── vlm/Qwen3-VL-Embedding-8B/...
│   ├── pixelrec/beauty/{best.pt,tokens_1024.pt}
│   ├── pixelrec/games/{best.pt,tokens_1024.pt}
│   ├── pixelrec/toys/{best.pt,tokens_1024.pt}
│   └── rcc/{beauty,games,toys}/compressor.pt
└── external_data/
    ├── beauty/images/...
    ├── games/images/...
    └── toys/images/...
```

`best.pt` and `tokens_1024.pt` are a pair. A checkpoint cannot reproduce its results with a cache generated from a different dataset, VLM configuration, or RCC model.

The loader accepts both the release-format checkpoint and the original research checkpoint. Original aggregator state-dict keys are mapped to the public `pixelrec_token_aggregator` name in memory; the public model source and runtime configuration remain PixelRec-only.

## 1. Integrity Verification

The weight release should include a `SHA256SUMS` file. Run this command from the repository root:

```bash
sha256sum --check weights/SHA256SUMS
```

For a VLM-only checksum file whose paths are relative to the VLM directory:

```bash
python verify_weights.py \
  --vlm-dir weights/vlm/Qwen3-VL-Embedding-8B \
  --sha256-manifest /path/to/VLM_SHA256SUMS
```

## 2. VLM Structure Verification

Check the Qwen configuration, tokenizer/processor files, hidden size, weight index, and every listed safetensors shard:

```bash
python verify_weights.py --vlm-dir weights/vlm/Qwen3-VL-Embedding-8B
```

A successful result reports `hidden_size: 4096` and the complete weight-shard list.

## 3. VLM Functional Verification

First render one product card, then perform a one-item extraction:

```bash
python preprocess.py \
  --dataset beauty \
  --images-dir external_data/beauty/images \
  --output-dir artifacts/beauty/preprocess \
  --limit 1

python extract_vlm.py \
  --data_dir data/beauty \
  --data_name beauty \
  --image_dir artifacts/beauty/preprocess/posters \
  --meta_json data/beauty/metadata.json \
  --model_name_or_path weights/vlm/Qwen3-VL-Embedding-8B \
  --output_path artifacts/beauty/vlm_smoke/tokens_4096.pt \
  --output_meta_json artifacts/beauty/vlm_smoke/tokens_4096.meta.json \
  --instruction "" \
  --max_token_cache_len 34 \
  --pooled_grid_h 8 \
  --pooled_grid_w 4 \
  --batch_size 1 \
  --dtype bfloat16 \
  --attn_implementation sdpa \
  --device cuda \
  --limit 1
```

The metadata must report:

```json
{
  "plan": "avg_pool_start_end_noprompt",
  "embedded_count": 1,
  "error_count": 0,
  "token_shape": [12102, 34, 4096],
  "hidden_dim": 4096
}
```

The first dimension remains the full Beauty item vocabulary plus padding; only the selected smoke-test row has a valid mask.

## 4. PixelRec Checkpoint Effect Verification

Run full-sort evaluation with each downloaded checkpoint/cache pair:

```bash
python verify_weights.py \
  --dataset beauty \
  --cache weights/pixelrec/beauty/tokens_1024.pt \
  --checkpoint weights/pixelrec/beauty/best.pt \
  --device cuda

python verify_weights.py \
  --dataset games \
  --cache weights/pixelrec/games/tokens_1024.pt \
  --checkpoint weights/pixelrec/games/best.pt \
  --device cuda

python verify_weights.py \
  --dataset toys \
  --cache weights/pixelrec/toys/tokens_1024.pt \
  --checkpoint weights/pixelrec/toys/best.pt \
  --device cuda
```

The verifier compares all six metrics against `configs/<dataset>.json` with a default absolute tolerance of `0.0005`. It exits with an error if the checkpoint belongs to another dataset, the cache shape is incompatible, loading fails, or any metric is outside tolerance.

Expected metrics:

| Dataset | H@5 | H@10 | H@20 | N@5 | N@10 | N@20 |
|---|---:|---:|---:|---:|---:|---:|
| Beauty | 0.0850 | 0.1194 | 0.1647 | 0.0593 | 0.0703 | 0.0817 |
| Games | 0.0994 | 0.1499 | 0.2210 | 0.0664 | 0.0825 | 0.1004 |
| Toys | 0.0909 | 0.1274 | 0.1736 | 0.0628 | 0.0746 | 0.0863 |
