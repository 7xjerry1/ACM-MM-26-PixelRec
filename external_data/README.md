# External Dataset Assets

Large downloaded product images are ignored by Git. The recommended layout is:

```text
external_data/
├── beauty/images/
├── games/images/
└── toys/images/
```

Image files may use either the internal item ID or the Amazon ASIN as their filename. The exact interaction sequences and compact item catalogs are already versioned under `data/`.
