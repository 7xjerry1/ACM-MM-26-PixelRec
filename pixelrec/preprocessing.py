import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from .config import load_dataset_config, sha256_file, verify_bundled_data
from .data import load_catalog


CANVAS_SIZE = (384, 544)
IMAGE_AREA_HEIGHT = 384
TEXT_MARGIN_X = 16
TEXT_MARGIN_TOP = 16
TEXT_FONT_SIZE = 22
POSTER_FONT_SIZE = 28
JPEG_QUALITY = 95
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
MEASURE_DRAW = ImageDraw.Draw(Image.new("RGB", (1, 1), (255, 255, 255)))


def load_font(size):
    candidates = (
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size), candidate
        except OSError:
            pass
    return ImageFont.load_default(), "Pillow-default"


def index_images(images_dir):
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {images_dir}")
    indexed = {}
    for path in sorted(images_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            indexed.setdefault(path.stem, path)
    return indexed


def resolve_image(item, indexed_images):
    return indexed_images.get(str(item["item_id"])) or indexed_images.get(item["asin"])


def text_width(text, font):
    if not text:
        return 0
    box = MEASURE_DRAW.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def line_height(font):
    box = MEASURE_DRAW.textbbox((0, 0), "Ag", font=font)
    return box[3] - box[1]


def max_fitting_prefix_len(text, max_width, font):
    if text_width(text, font) <= max_width:
        return len(text)
    low, high, best = 1, len(text), 1
    while low <= high:
        middle = (low + high) // 2
        if text_width(text[:middle], font) <= max_width:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    return best


def split_one_line(text, max_width, font):
    if text_width(text, font) <= max_width:
        return text, ""
    fit_length = max_fitting_prefix_len(text, max_width, font)
    segment = text[:fit_length]
    break_index = max((index for index, char in enumerate(segment) if char.isspace()), default=-1)
    if break_index > 0:
        next_index = break_index + 1
        while next_index < len(text) and text[next_index].isspace():
            next_index += 1
        return segment[:break_index], text[next_index:]
    return segment, text[fit_length:]


def truncate_with_ellipsis(text, max_width, font):
    ellipsis = "..."
    if text_width(text, font) <= max_width:
        return text
    if text_width(ellipsis, font) >= max_width:
        return ellipsis
    low, high, best = 0, len(text), ""
    while low <= high:
        middle = (low + high) // 2
        candidate = text[:middle] + ellipsis
        if text_width(candidate, font) <= max_width:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best or ellipsis


def wrap_text(text, max_width, font, max_lines):
    remaining = text
    lines = []
    for _ in range(max_lines - 1):
        if not remaining:
            break
        if text_width(remaining, font) <= max_width:
            lines.append(remaining)
            remaining = ""
            break
        line, remaining = split_one_line(remaining, max_width, font)
        lines.append(line)
    if remaining:
        lines.append(
            remaining if text_width(remaining, font) <= max_width else truncate_with_ellipsis(remaining, max_width, font)
        )
    return lines[:max_lines]


def paste_contain_image(canvas, image):
    source_width, source_height = image.size
    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"Invalid source image size: {image.size}")
    scale = min(CANVAS_SIZE[0] / source_width, IMAGE_AREA_HEIGHT / source_height)
    width = max(1, int(round(source_width * scale)))
    height = max(1, int(round(source_height * scale)))
    resized = image.resize((width, height), Image.Resampling.LANCZOS)
    canvas.paste(resized, ((CANVAS_SIZE[0] - width) // 2, (IMAGE_AREA_HEIGHT - height) // 2))


def draw_lines(draw, lines, font, start_x, start_y, spacing, centered=False):
    height = line_height(font)
    for line in lines:
        x = (CANVAS_SIZE[0] - text_width(line, font)) // 2 if centered else start_x
        draw.text((x, start_y), line, font=font, fill=(0, 0, 0))
        start_y += height + spacing


def render_with_image(source, display_title, title_font):
    canvas = Image.new("RGB", CANVAS_SIZE, (255, 255, 255))
    paste_contain_image(canvas, source)
    draw = ImageDraw.Draw(canvas)
    draw.line((0, IMAGE_AREA_HEIGHT, CANVAS_SIZE[0], IMAGE_AREA_HEIGHT), fill=(220, 220, 220), width=1)
    lines = wrap_text(display_title, CANVAS_SIZE[0] - 2 * TEXT_MARGIN_X, title_font, 4)
    draw_lines(draw, lines, title_font, TEXT_MARGIN_X, IMAGE_AREA_HEIGHT + TEXT_MARGIN_TOP, 6)
    return canvas


def render_title_poster(display_title, poster_font):
    canvas = Image.new("RGB", CANVAS_SIZE, (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    lines = wrap_text(display_title, CANVAS_SIZE[0] - 2 * TEXT_MARGIN_X, poster_font, 8)
    total_height = line_height(poster_font) * len(lines) + 8 * max(0, len(lines) - 1)
    draw_lines(draw, lines, poster_font, TEXT_MARGIN_X, max(0, (CANVAS_SIZE[1] - total_height) // 2), 8, centered=True)
    return canvas


def render_poster(item, image_path, output_path, title_font, poster_font):
    title = item["title"].strip()
    display_title = title or f"ASIN: {item['asin']}"
    status = "rendered_with_image"
    if image_path is not None:
        try:
            with Image.open(image_path) as source:
                canvas = render_with_image(source.convert("RGB"), display_title, title_font)
        except (OSError, UnidentifiedImageError):
            image_path = None
            status = "rendered_title_only_image_error"
    else:
        status = "rendered_title_only_no_image"

    if image_path is None:
        canvas = render_title_poster(display_title, poster_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="JPEG", quality=JPEG_QUALITY)
    if not title:
        status += "_asin_fallback"
    return status


def preprocess_dataset(dataset, images_dir, output_dir, force=False, limit=None):
    config = load_dataset_config(dataset)
    paths = verify_bundled_data(dataset)
    by_id, _ = load_catalog(paths["catalog"])
    indexed_images = index_images(images_dir)
    output_dir = Path(output_dir)
    poster_dir = output_dir / "posters"
    title_font, title_font_path = load_font(TEXT_FONT_SIZE)
    poster_font, poster_font_path = load_font(POSTER_FONT_SIZE)
    counts = Counter()
    records = []
    item_ids = sorted(by_id)
    if limit is not None:
        item_ids = item_ids[: int(limit)]
    for item_id in item_ids:
        item = by_id[item_id]
        image_path = resolve_image(item, indexed_images)
        output_path = poster_dir / f"{item_id}.jpg"
        if output_path.exists() and not force:
            status = "reused_existing"
        else:
            status = render_poster(item, image_path, output_path, title_font, poster_font)
        counts[status] += 1
        records.append(
            {
                "item_id": item_id,
                "asin": item["asin"],
                "source_image": str(image_path) if image_path else None,
                "poster": str(output_path.relative_to(output_dir)),
                "status": status,
            }
        )
    missing_outputs = [record["item_id"] for record in records if not (output_dir / record["poster"]).is_file()]
    if missing_outputs:
        raise RuntimeError(f"Poster generation failed for item IDs: {missing_outputs[:20]}")
    manifest = {
        "pipeline": "PixelRec dataset catalog + downloaded images -> title-concatenated posters",
        "dataset": config["dataset"],
        "num_items": len(item_ids),
        "full_dataset_items": int(config["num_items"]),
        "canvas_size": list(CANVAS_SIZE),
        "image_area_size": [CANVAS_SIZE[0], IMAGE_AREA_HEIGHT],
        "text_area_size": [CANVAS_SIZE[0], CANVAS_SIZE[1] - IMAGE_AREA_HEIGHT],
        "text_font_size": TEXT_FONT_SIZE,
        "poster_font_size": POSTER_FONT_SIZE,
        "title_font": title_font_path,
        "poster_font": poster_font_path,
        "jpeg_quality": JPEG_QUALITY,
        "sequence_sha256": sha256_file(paths["sequence"]),
        "catalog_sha256": sha256_file(paths["catalog"]),
        "status_counts": dict(sorted(counts.items())),
        "items": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "preprocess_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest_path


def parse_args():
    parser = argparse.ArgumentParser(description="Render PixelRec product posters from bundled catalogs and downloaded images.")
    parser.add_argument("--dataset", required=True, choices=("beauty", "games", "toys"))
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    path = preprocess_dataset(args.dataset, args.images_dir, args.output_dir, force=args.force, limit=args.limit)
    print(path)


if __name__ == "__main__":
    main()
