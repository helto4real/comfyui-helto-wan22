import json
from dataclasses import dataclass


@dataclass(frozen=True)
class FolderConfig:
    alias: str
    path: str
    enabled: bool = True


@dataclass(frozen=True)
class GuideItem:
    folder_alias: str
    filename: str
    position: float
    calculated_frame: int
    strength: float = 1.0
    label: str = ""
    enabled: bool = True


def parse_guides_json(raw):
    if not raw:
        return []
    if isinstance(raw, dict):
        data = raw
    else:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid guides_json: {exc}") from exc

    guides = data.get("guides", data if isinstance(data, list) else [])
    parsed = []
    for item in guides:
        if not isinstance(item, dict):
            continue
        folder_alias = str(item.get("folder_alias", item.get("folderAlias", ""))).strip()
        filename = str(item.get("filename", "")).strip()
        if not folder_alias or not filename:
            continue
        parsed.append(
            GuideItem(
                folder_alias=folder_alias,
                filename=filename,
                position=float(item.get("position", 0)),
                calculated_frame=int(item.get("calculated_frame", item.get("calculatedFrame", 0))),
                strength=float(item.get("strength", 1.0)),
                label=str(item.get("label", "")),
                enabled=bool(item.get("enabled", True)),
            )
        )
    return parsed


def guide_summary(guides):
    enabled = [guide for guide in guides if guide.enabled]
    if not enabled:
        return "No guides"
    pieces = [f"{guide.calculated_frame}f {guide.filename}" for guide in enabled[:4]]
    suffix = "" if len(enabled) <= 4 else f", +{len(enabled) - 4} more"
    return f"{len(enabled)} guides: {', '.join(pieces)}{suffix}"
