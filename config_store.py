import json
import os
import re
from pathlib import Path

import folder_paths

from .guide_models import FolderConfig

NODE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = NODE_DIR / "config"
GUIDE_SETS_DIR = CONFIG_DIR / "guide_sets"
FOLDERS_FILE = CONFIG_DIR / "folders.json"
THUMB_CACHE_DIR = NODE_DIR / "thumbnail_cache"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def ensure_dirs():
    CONFIG_DIR.mkdir(exist_ok=True)
    GUIDE_SETS_DIR.mkdir(exist_ok=True)
    THUMB_CACHE_DIR.mkdir(exist_ok=True)


def _default_folder():
    return FolderConfig(alias="input", path=os.path.normpath(folder_paths.get_input_directory()), enabled=True)


def _safe_alias(alias):
    alias = str(alias or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_. -]{1,80}", alias):
        raise ValueError("Alias must be 1-80 characters using letters, numbers, spaces, dot, underscore, or dash.")
    return alias


def load_folders():
    ensure_dirs()
    default = _default_folder()
    if not FOLDERS_FILE.exists():
        return [default]
    try:
        data = json.loads(FOLDERS_FILE.read_text(encoding="utf-8") or "{}")
    except Exception:
        return [default]

    folders = []
    seen = set()
    for entry in data.get("folders", []):
        try:
            alias = _safe_alias(entry.get("alias"))
        except ValueError:
            continue
        path = os.path.normpath(os.path.expanduser(str(entry.get("path", ""))))
        if alias in seen or not path:
            continue
        folders.append(FolderConfig(alias=alias, path=path, enabled=bool(entry.get("enabled", True))))
        seen.add(alias)

    if default.alias not in seen:
        folders.insert(0, default)
    return folders


def save_folders(folders):
    ensure_dirs()
    payload = {
        "version": 1,
        "folders": [
            {"alias": folder.alias, "path": os.path.normpath(folder.path), "enabled": folder.enabled}
            for folder in folders
        ],
    }
    FOLDERS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def folder_by_alias(alias):
    for folder in load_folders():
        if folder.alias == alias:
            return folder
    raise ValueError(f"Unknown folder alias: {alias}")


def resolve_image_path(alias, filename):
    folder = folder_by_alias(alias)
    if not folder.enabled:
        raise ValueError(f"Folder alias is disabled: {alias}")
    root = Path(os.path.normpath(os.path.expanduser(folder.path))).resolve()
    candidate = (root / filename).resolve()
    if root != candidate and root not in candidate.parents:
        raise ValueError("Invalid image path.")
    if candidate.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {candidate.suffix}")
    if not candidate.is_file():
        raise FileNotFoundError(f"Image not found: {alias}/{filename}")
    return candidate


def add_folder(alias, path):
    alias = _safe_alias(alias)
    path = os.path.normpath(os.path.expanduser(str(path or "")))
    if not os.path.isdir(path):
        raise ValueError(f"Folder does not exist: {path}")
    folders = load_folders()
    if any(folder.alias == alias for folder in folders):
        raise ValueError(f"Folder alias already exists: {alias}")
    folders.append(FolderConfig(alias=alias, path=path, enabled=True))
    save_folders(folders)
    return folders


def remove_folder(alias):
    if alias == "input":
        raise ValueError("Cannot remove the default input folder.")
    folders = load_folders()
    next_folders = [folder for folder in folders if folder.alias != alias]
    if len(next_folders) == len(folders):
        raise ValueError(f"Folder alias not found: {alias}")
    save_folders(next_folders)
    return next_folders


def safe_guide_set_name(name):
    name = str(name or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_. -]{1,100}", name):
        raise ValueError("Guide set name must be 1-100 safe filename characters.")
    return name


def guide_set_path(name):
    ensure_dirs()
    return GUIDE_SETS_DIR / f"{safe_guide_set_name(name)}.json"
