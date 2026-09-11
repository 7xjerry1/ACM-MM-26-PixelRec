# Local Weight Directory

Large model files are intentionally ignored by Git. Place downloaded weights in this structure:

```text
weights/
├── vlm/
│   └── Qwen3-VL-Embedding-8B/
│       ├── config.json
│       ├── model.safetensors.index.json
│       ├── model-00001-of-....safetensors
│       ├── tokenizer.json
│       └── preprocessor_config.json
├── pixelrec/
│   ├── beauty/
│   │   ├── best.pt
│   │   └── tokens_1024.pt
│   ├── games/
│   │   ├── best.pt
│   │   └── tokens_1024.pt
│   └── toys/
│       ├── best.pt
│       └── tokens_1024.pt
└── rcc/
    ├── beauty/compressor.pt
    ├── games/compressor.pt
    └── toys/compressor.pt
```

`tokens_1024.pt` is required together with `best.pt`, because PixelRec aggregates the dataset-specific compressed VLM token cache at both training and inference time.

See `WEIGHTS.md` in the repository root for download, integrity, functional, and benchmark verification instructions.
