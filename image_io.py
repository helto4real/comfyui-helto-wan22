import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from .config_store import IMAGE_EXTENSIONS, THUMB_CACHE_DIR


def parse_pad_color(value):
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return tuple(max(0, min(255, int(v))) for v in value[:3])
    text = str(value or "0,0,0").strip()
    if text.startswith("#"):
        text = text[1:]
        if len(text) == 6:
            return tuple(int(text[i : i + 2], 16) for i in (0, 2, 4))
    parts = [part.strip() for part in text.replace(" ", ",").split(",") if part.strip()]
    if len(parts) == 3:
        return tuple(max(0, min(255, int(float(part)))) for part in parts)
    return (0, 0, 0)


def load_rgb_image(path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")


def resize_image(image, width, height, mode="contain", pad_color="0,0,0"):
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive.")
    mode = mode or "contain"

    if mode == "stretch":
        return image.resize((width, height), Image.Resampling.LANCZOS)

    if mode == "crop":
        src_w, src_h = image.size
        scale = max(width / src_w, height / src_h)
        new_size = (max(1, round(src_w * scale)), max(1, round(src_h * scale)))
        resized = image.resize(new_size, Image.Resampling.LANCZOS)
        left = (resized.width - width) // 2
        top = (resized.height - height) // 2
        return resized.crop((left, top, left + width, top + height))

    src_w, src_h = image.size
    scale = min(width / src_w, height / src_h)
    new_size = (max(1, round(src_w * scale)), max(1, round(src_h * scale)))
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), parse_pad_color(pad_color))
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return canvas


def image_to_tensor(image):
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def tensor_to_image(tensor):
    array = tensor[:, :, :3].detach().cpu().numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def resize_tensor_images(images, width, height, resize_mode, pad_color):
    frames = []
    for frame in images:
        image = tensor_to_image(frame)
        resized = resize_image(image, width, height, resize_mode, pad_color)
        frames.append(image_to_tensor(resized)[0])
    if not frames:
        raise ValueError("Image sequence is empty.")
    return torch.stack(frames)


def load_guide_tensor(path, width, height, resize_mode, pad_color):
    image = load_rgb_image(path)
    original_size = image.size
    resized = resize_image(image, width, height, resize_mode, pad_color)
    return image_to_tensor(resized), original_size


def list_images(root, recursive=True):
    root = Path(root)
    results = []
    if not root.is_dir():
        return results
    walker = os.walk(root) if recursive else [(root, [], [p.name for p in root.iterdir() if p.is_file()])]
    for dirpath, _, filenames in walker:
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            rel = path.relative_to(root).as_posix()
            width = height = 0
            try:
                with Image.open(path) as image:
                    width, height = image.size
            except Exception:
                pass
            results.append(
                {
                    "filename": rel,
                    "width": width,
                    "height": height,
                    "mtime": path.stat().st_mtime if path.exists() else 0,
                }
            )
    return sorted(results, key=lambda item: item["filename"].lower())


def thumbnail_path(image_path, max_size=320):
    image_path = Path(image_path)
    mtime = image_path.stat().st_mtime if image_path.exists() else 0
    key = hashlib.sha256(f"{image_path}:{mtime}:{max_size}".encode("utf-8")).hexdigest()
    return THUMB_CACHE_DIR / f"{key}.webp"


def make_thumbnail(image_path, max_size=320):
    THUMB_CACHE_DIR.mkdir(exist_ok=True)
    out = thumbnail_path(image_path, max_size)
    if out.exists():
        return out
    image = load_rgb_image(image_path)
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    image.save(out, "WEBP", quality=90, method=4)
    return out
