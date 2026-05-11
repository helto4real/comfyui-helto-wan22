import json
import os
import urllib.parse

from aiohttp import web
import server

from .config_store import (
    IMAGE_EXTENSIONS,
    add_folder,
    folder_by_alias,
    guide_set_path,
    load_folders,
    remove_folder,
    safe_guide_set_name,
)
from .image_io import list_images, make_thumbnail


def _folder_payload():
    folders = []
    for folder in load_folders():
        exists = os.path.isdir(folder.path)
        folders.append(
            {
                "alias": folder.alias,
                "enabled": folder.enabled,
                "exists": exists,
                "image_count": len(list_images(folder.path)) if exists else 0,
            }
        )
    return folders


@server.PromptServer.instance.routes.get("/wan22_guides/folders")
async def get_folders(request):
    return web.json_response({"folders": _folder_payload()})


@server.PromptServer.instance.routes.post("/wan22_guides/folders")
async def post_folder(request):
    try:
        data = await request.json()
        add_folder(data.get("alias"), data.get("path"))
        return web.json_response({"status": "ok", "folders": _folder_payload()})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)


@server.PromptServer.instance.routes.delete("/wan22_guides/folders")
async def delete_folder(request):
    try:
        alias = request.query.get("alias", "")
        remove_folder(alias)
        return web.json_response({"status": "ok", "folders": _folder_payload()})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)


@server.PromptServer.instance.routes.get("/wan22_guides/images")
async def get_images(request):
    try:
        alias = request.query.get("alias", "")
        recursive = request.query.get("recursive", "1").lower() not in {"0", "false", "no"}
        folder = folder_by_alias(alias)
        if not os.path.isdir(folder.path):
            return web.json_response({"images": [], "warning": "Folder does not exist."})
        images = list_images(folder.path, recursive=recursive)
        for image in images:
            image["thumb_url"] = (
                "/wan22_guides/thumb?"
                + urllib.parse.urlencode({"alias": alias, "filename": image["filename"], "t": int(image["mtime"])})
            )
        return web.json_response({"images": images})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)


@server.PromptServer.instance.routes.post("/wan22_guides/refresh")
async def refresh(request):
    return web.json_response({"status": "ok", "folders": _folder_payload()})


@server.PromptServer.instance.routes.get("/wan22_guides/thumb")
async def get_thumb(request):
    try:
        alias = request.query.get("alias", "")
        filename = urllib.parse.unquote(request.query.get("filename", ""))
        folder = folder_by_alias(alias)
        root = os.path.realpath(folder.path)
        path = os.path.realpath(os.path.join(root, filename))
        if root != path and not path.startswith(root + os.sep):
            return web.Response(status=403, text="Invalid path")
        if not os.path.isfile(path):
            return web.Response(status=404, text="Image not found")
        thumb = make_thumbnail(path)
        return web.FileResponse(thumb, headers={"Cache-Control": "public, max-age=86400", "Content-Type": "image/webp"})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)


@server.PromptServer.instance.routes.get("/wan22_guides/image")
async def get_image(request):
    try:
        alias = request.query.get("alias", "")
        filename = urllib.parse.unquote(request.query.get("filename", ""))
        folder = folder_by_alias(alias)
        root = os.path.realpath(folder.path)
        path = os.path.realpath(os.path.join(root, filename))
        if root != path and not path.startswith(root + os.sep):
            return web.Response(status=403, text="Invalid path")
        if os.path.splitext(path)[1].lower() not in IMAGE_EXTENSIONS:
            return web.Response(status=400, text="Unsupported image type")
        if not os.path.isfile(path):
            return web.Response(status=404, text="Image not found")
        return web.FileResponse(path, headers={"Cache-Control": "private, max-age=300"})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)


@server.PromptServer.instance.routes.get("/wan22_guides/guide_sets")
async def list_guide_sets(request):
    path = guide_set_path("placeholder").parent
    sets = sorted(p.stem for p in path.glob("*.json"))
    return web.json_response({"guide_sets": sets})


@server.PromptServer.instance.routes.get("/wan22_guides/guide_sets/{name}")
async def load_guide_set(request):
    try:
        path = guide_set_path(request.match_info["name"])
        if not path.exists():
            return web.json_response({"error": "Guide set not found."}, status=404)
        return web.json_response(json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)


@server.PromptServer.instance.routes.post("/wan22_guides/guide_sets/{name}")
async def save_guide_set(request):
    try:
        name = safe_guide_set_name(request.match_info["name"])
        data = await request.json()
        data["version"] = int(data.get("version", 1))
        guide_set_path(name).write_text(json.dumps(data, indent=2), encoding="utf-8")
        return web.json_response({"status": "ok", "name": name})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)


@server.PromptServer.instance.routes.delete("/wan22_guides/guide_sets/{name}")
async def delete_guide_set(request):
    try:
        path = guide_set_path(request.match_info["name"])
        if path.exists():
            path.unlink()
        return web.json_response({"status": "ok"})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)
