# PixelRec: From Modality-Level Fusion to Signal-Patch-Level Fusion for Multimodal Sequential Recommendation

Official PyTorch implementation of **PixelRec**, by Ruijie Xiao, Bo Yang (corresponding author), and Guipeng Xv. 

PixelRec revisits Multimodal Sequential Recommendation (MMSR) by moving beyond coarse modality-level fusion. It renders product text together with the original image as a unified visual input, encodes that input into fine-grained signal patches, and adaptively fuses the patches with a Vision-Recommendation Aggregator. A Re-Construction Compression (RCC) module reduces the GPU cost of processing these features. Across extensive experiments, PixelRec improves recommendation accuracy by 4.6%–15.9% over state-of-the-art MMSR methods.

## Code Overview

PixelRec is a standalone PyTorch implementation of a pixel-first sequential recommender. It renders each product image and title into one product card, extracts token-level visual states with Qwen3-VL-Embedding-8B, compresses those states with a reconstruction-based linear compressor (RCC), and trains a lightweight query aggregator with a causal sequential recommendation backbone.

This repository contains the exact Beauty, Video Games, and Toys interaction splits used for the reported results. Product images, Qwen weights, generated token caches, compressor checkpoints, and recommendation checkpoints are intentionally not included.

## Pipeline

```text
bundled sequences + bundled catalog + downloaded product images
    -> product cards
    -> Qwen3-VL token cache
    -> RCC token cache
    -> PixelRec training
    -> evaluation
```

## Repository Layout

```text
PixelRec/
├── configs/                 # Reproducible dataset and hyperparameter configs
├── data/                    # Exact sequences, metadata, and compact item catalogs
├── external_data/           # Local downloaded image archives (ignored by Git)
├── weights/                 # Local VLM/RCC/PixelRec weights (ignored by Git)
├── pixelrec/
│   ├── model/PixelRec.py    # PixelRec model and token aggregator
│   ├── preprocessing.py     # Product-card renderer
│   ├── vlm.py               # Qwen3-VL token extraction
│   ├── compression.py       # RCC training and cache export
│   ├── training.py          # Training entry point
│   ├── evaluation.py        # Full-sort test evaluation
│   └── recommendation.py    # Item-ID/ASIN Top-K inference
├── results/benchmarks.json
└── run_pipeline.py          # Resumable end-to-end runner
```

Generated files are placed under `artifacts/<dataset>/` and ignored by Git.

## Environment

The tested environment uses Python 3.10/3.11, PyTorch 2.6.0 with CUDA 12.4, Transformers 4.57.1, and ModelScope 1.34.0. Install the PyTorch build appropriate for your CUDA installation first if the default wheel is unsuitable, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Downloaded Weights and Large Dataset Assets

Put the complete VLM directory at:

```text
weights/vlm/Qwen3-VL-Embedding-8B/
```

Put released PixelRec checkpoint/cache pairs at:

```text
weights/pixelrec/beauty/{best.pt,tokens_1024.pt}
weights/pixelrec/games/{best.pt,tokens_1024.pt}
weights/pixelrec/toys/{best.pt,tokens_1024.pt}
```

Put the downloaded product images at `external_data/<dataset>/images/`. These directories are ignored by Git.

Release downloads:

- VLM: [official Qwen3-VL-Embedding-8B page](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B)
- PixelRec/RCC weights and compressed caches: [Google Drive](https://drive.google.com/drive/folders/1ENW80IJidpnzwDHIi9cfgFbIZjKBNm56)
- Beauty/Games/Toys image archives: [Google Drive](https://drive.google.com/drive/folders/1vWV8_0sqAoLvLDzjNEZ7GEV5pDJbaUXr)

The two Google Drive downloads are split ZIP archives. Download every numbered part, the final `.zip` file, and `SHA256SUMS` from the corresponding folder. Verify, reconstruct, and extract them from the directory containing the downloaded parts:

```bash
# PixelRec/RCC weights and compressed caches
sha256sum --check SHA256SUMS
zip -s 0 pixelrec_weights.zip --out pixelrec_weights_full.zip
unzip pixelrec_weights_full.zip

# Beauty/Games/Toys image archives
sha256sum --check SHA256SUMS
zip -s 0 pixelrec_datasets.zip --out pixelrec_datasets_full.zip
unzip pixelrec_datasets_full.zip
```

See [WEIGHTS.md](WEIGHTS.md) for the exact archive layout, SHA-256 checks, a one-item VLM functional test, and full benchmark verification for all three PixelRec checkpoints.

## Bundled Datasets

| Dataset | Users | Items | Interactions | Sequence file |
|---|---:|---:|---:|---|
| Beauty | 22,363 | 12,101 | 198,502 | `data/beauty/beauty.txt` |
| Video Games | 24,303 | 10,672 | 231,780 | `data/games/games.txt` |
| Toys and Games | 19,412 | 11,924 | 167,597 | `data/toys/toys.txt` |

Every line in a sequence file is one chronologically ordered user history. Item IDs start at 1; ID 0 is reserved for padding. `items.jsonl` maps each item ID to its ASIN and title. The Games catalog contains 4,667 items without a recovered title; preprocessing uses `ASIN: <value>` for those items, matching the reference product-card fallback.

Verify all committed files before running an experiment:

```bash
python verify_data.py
```

## Image Input

Prepare one directory per dataset containing the already-downloaded product images. Images may be named with either the internal item ID or ASIN, for example:

```text
images/1.jpg
images/B00004TMFE.png
```

Supported extensions are JPG, JPEG, PNG, WebP, and BMP. When both forms exist, the item-ID filename takes precedence. Missing or unreadable images produce a white title card; missing titles fall back to the ASIN. Preprocessing always produces `<item_id>.jpg` posters.

## End-to-End Run

Use a local Qwen checkpoint:

```bash
python run_pipeline.py \
  --dataset beauty \
  --images-dir external_data/beauty/images \
  --qwen-model weights/vlm/Qwen3-VL-Embedding-8B \
  --gpu-device cuda \
  --device cuda \
  --stages all
```

When `weights/vlm/Qwen3-VL-Embedding-8B/` exists, `--qwen-model` may be omitted because the pipeline discovers that standard location automatically.

Or let ModelScope obtain the model:

```bash
python run_pipeline.py \
  --dataset beauty \
  --images-dir /path/to/beauty/images \
  --modelscope-id Qwen/Qwen3-VL-Embedding-8B \
  --stages all
```

`all` runs preprocessing, VLM extraction, compression, training, and evaluation. Existing valid caches and checkpoints are reused. Repeat `--force-stage` to rebuild selected stages:

```bash
python run_pipeline.py ... --force-stage vlm --force-stage compress --stages vlm,compress,train,evaluate
```

## Run Individual Stages

Render product cards:

```bash
python preprocess.py \
  --dataset beauty \
  --images-dir /path/to/beauty/images \
  --output-dir artifacts/beauty/preprocess
```

Extract the reference VLM cache:

```bash
python extract_vlm.py \
  --data_dir data/beauty \
  --data_name beauty \
  --image_dir artifacts/beauty/preprocess/posters \
  --meta_json data/beauty/metadata.json \
  --model_name_or_path /path/to/Qwen3-VL-Embedding-8B \
  --output_path artifacts/beauty/vlm/tokens_4096.pt \
  --output_meta_json artifacts/beauty/vlm/tokens_4096.meta.json \
  --instruction "" \
  --max_token_cache_len 34 \
  --pooled_grid_h 8 \
  --pooled_grid_w 4 \
  --batch_size 1 \
  --dtype bfloat16 \
  --attn_implementation sdpa \
  --device cuda
```

Train RCC and export the compressed cache:

```bash
python compress.py \
  --input_path artifacts/beauty/vlm/tokens_4096.pt \
  --output_path artifacts/beauty/compression/tokens_1024.pt \
  --checkpoint_path artifacts/beauty/compression/compressor.pt \
  --output_meta_json artifacts/beauty/compression/tokens_1024.meta.json \
  --input_dim 4096 \
  --bottleneck_dim 1024 \
  --lambda1 1.0 \
  --lambda2 0.1 \
  --batch_size 2048 \
  --epochs 15 \
  --train_tokens_per_epoch 524288 \
  --lr 0.001 \
  --seed 2026 \
  --device cuda
```

Train PixelRec:

```bash
python train.py \
  --dataset beauty \
  --cache artifacts/beauty/compression/tokens_1024.pt \
  --output-dir artifacts/beauty/training \
  --device cuda
```

Evaluate a checkpoint on the complete test split:

```bash
python evaluate.py \
  --dataset beauty \
  --cache artifacts/beauty/compression/tokens_1024.pt \
  --checkpoint artifacts/beauty/training/best.pt \
  --output artifacts/beauty/evaluation.json \
  --device cuda
```

Generate recommendations from a mixture of item IDs and ASINs:

```bash
python recommend.py \
  --dataset beauty \
  --cache artifacts/beauty/compression/tokens_1024.pt \
  --checkpoint artifacts/beauty/training/best.pt \
  --history '1,B00004TMFE,42' \
  --top-k 20 \
  --output artifacts/beauty/recommendations.json \
  --device cuda
```

The most recent 50 history items are used. Padding, unknown items, and already-seen items are never returned; unknown input IDs or ASINs are reported as errors.


## Reference Results

| Dataset | H@5 | H@10 | H@20 | N@5 | N@10 | N@20 |
|---|---:|---:|---:|---:|---:|---:|
| Beauty | 0.0850 | 0.1194 | 0.1647 | 0.0593 | 0.0703 | 0.0817 |
| Games | 0.0994 | 0.1499 | 0.2210 | 0.0664 | 0.0825 | 0.1004 |
| Toys | 0.0909 | 0.1274 | 0.1736 | 0.0628 | 0.0746 | 0.0863 |

The protocol is leave-one-out full-sort evaluation. The final item is the test target, the penultimate item is the validation target, and previously interacted items are masked before ranking.

## Tests

Run the CPU-safe unit and smoke tests:

```bash
python -m unittest discover -s tests -v
python -m py_compile *.py pixelrec/*.py pixelrec/model/*.py
```

Use `--limit` with preprocessing/VLM stages for a small GPU extraction smoke test. A limited cache is intended only for interface validation, not benchmark training.

## Citation and Notices

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream attribution and license text. This repository does not include a project-level license.
