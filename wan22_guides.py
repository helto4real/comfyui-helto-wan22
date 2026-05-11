import torch

from .config_store import resolve_image_path
from .guide_models import parse_guides_json
from .image_io import load_guide_tensor, resize_tensor_images


def clamp01(value):
    return max(0.0, min(1.0, float(value)))


def resolve_timing(guides, timing_mode, fps, length, duplicate_policy):
    resolved = []
    occupied = {}
    length = max(1, int(length))
    for guide in guides:
        if not guide.enabled:
            continue
        frame = round(guide.position * fps) if timing_mode == "seconds" else int(round(guide.position))
        if frame < 0:
            frame = length + frame
        frame = max(0, min(length - 1, frame))
        if frame in occupied:
            if duplicate_policy == "error":
                raise ValueError(f"Duplicate guide frame {frame}: {guide.filename}")
            if duplicate_policy == "keep_first":
                continue
            if duplicate_policy == "keep_last":
                prior = occupied[frame]
                resolved[prior] = None
                occupied[frame] = len(resolved)
            if duplicate_policy == "offset_next":
                while frame in occupied and frame < length - 1:
                    frame += 1
                if frame in occupied:
                    raise ValueError(f"Could not offset duplicate guide near final frame for {guide.filename}")
                occupied[frame] = len(resolved)
        else:
            occupied[frame] = len(resolved)
        resolved.append((frame, guide))
    return sorted([item for item in resolved if item is not None], key=lambda item: item[0])


def wan_spatial_scale(vae):
    try:
        return int(vae.spacial_compression_encode())
    except Exception:
        return 16


def wan_latent_channels(vae):
    return int(getattr(vae, "latent_channels", 48))


def latent_length(length):
    return ((max(1, int(length)) - 1) // 4) + 1


def round_dimension(value, scale):
    rounded = (int(value) // int(scale)) * int(scale)
    if rounded <= 0:
        raise ValueError(f"Dimension {value} is smaller than the WAN spatial scale {scale}.")
    return rounded


def create_empty_latent(width, height, length, batch_size, vae, latent_channels=None):
    import comfy.model_management

    scale = wan_spatial_scale(vae)
    channels = int(latent_channels or wan_latent_channels(vae))
    samples = torch.zeros(
        [int(batch_size), channels, latent_length(length), int(height) // scale, int(width) // scale],
        device=comfy.model_management.intermediate_device(),
    )
    return {"samples": samples}


def create_wan22_concat_latent(batch_size, channels, slots, latent_t, latent_h, latent_w, device, dtype):
    import comfy.latent_formats

    concat_latent = torch.zeros([int(batch_size), channels, latent_t, latent_h, latent_w], device=device, dtype=dtype)
    if channels == 48:
        concat_latent = comfy.latent_formats.Wan22().process_out(concat_latent)
    else:
        concat_latent = comfy.latent_formats.Wan21().process_out(concat_latent)
    return concat_latent.repeat(1, int(slots), 1, 1, 1)


def fit_latent_channels(latent, channels):
    channels = int(channels)
    if latent.shape[1] == channels:
        return latent
    if latent.shape[1] > channels:
        return latent[:, :channels]
    padding = torch.zeros(
        latent.shape[0],
        channels - latent.shape[1],
        latent.shape[2],
        latent.shape[3],
        latent.shape[4],
        device=latent.device,
        dtype=latent.dtype,
    )
    return torch.cat((latent, padding), dim=1)


def upscale_images(images, width, height):
    import comfy.utils

    return comfy.utils.common_upscale(images.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)


def write_frames(timeline, mask_frames, images, start_frame, strength):
    if images is None or images.shape[0] <= 0:
        return
    start_frame = int(start_frame)
    end_frame = min(timeline.shape[0], start_frame + images.shape[0])
    if start_frame >= end_frame:
        return
    write_len = end_frame - start_frame
    timeline[start_frame:end_frame] = images[:write_len].to(device=timeline.device, dtype=timeline.dtype)
    mask_frames[:, :, start_frame:end_frame] = 1.0 - clamp01(strength)


def build_guide_timeline(
    vae,
    width,
    height,
    length,
    fps,
    timing_mode,
    resize_mode,
    duplicate_policy,
    pad_color,
    global_strength,
    guides_json,
    start_images=None,
    start_images_strength=0.85,
):
    import comfy.model_management

    guides = parse_guides_json(guides_json)
    resolved = resolve_timing(guides, timing_mode, float(fps), length, duplicate_policy)
    device = comfy.model_management.intermediate_device()
    timeline = torch.ones((int(length), int(height), int(width), 3), device=device, dtype=torch.float32) * 0.5
    mask_frames = torch.ones(
        (1, 1, latent_length(length) * 4, int(height) // wan_spatial_scale(vae), int(width) // wan_spatial_scale(vae)),
        device=device,
        dtype=torch.float32,
    )

    if start_images is not None:
        if start_images.shape[0] <= 0:
            raise ValueError("Start image sequence is empty.")
        start = resize_tensor_images(start_images[: int(length)], width, height, resize_mode, pad_color)
        start = start.to(device=device, dtype=torch.float32)
        sequence_len = min(start.shape[0], int(length))
        for frame_idx, guide in resolved:
            if frame_idx < sequence_len:
                raise ValueError(f"Manual guide {guide.filename} at frame {frame_idx} overlaps start_images.")
        write_frames(timeline, mask_frames, start, 0, clamp01(global_strength) * clamp01(start_images_strength))

    for frame_idx, guide in resolved:
        image_path = resolve_image_path(guide.folder_alias, guide.filename)
        image, _ = load_guide_tensor(image_path, width, height, resize_mode, pad_color)
        image = image.to(device=device, dtype=torch.float32)
        write_frames(timeline, mask_frames, image, frame_idx, clamp01(global_strength) * clamp01(guide.strength))

    return timeline, mask_frames


def apply_wan22_guides(
    positive,
    negative,
    vae,
    width,
    height,
    length,
    batch_size,
    fps,
    timing_mode,
    resize_mode,
    duplicate_policy,
    pad_color,
    global_strength,
    guides_json,
    start_images=None,
    start_images_strength=0.85,
    ref_image=None,
    control_video=None,
    sampler_latent_channels=None,
    concat_slots=2,
    concat_slot_channels=None,
):
    import node_helpers

    scale = wan_spatial_scale(vae)
    width = round_dimension(width, scale)
    height = round_dimension(height, scale)
    length = max(1, int(length))
    batch_size = max(1, int(batch_size))
    vae_channels = wan_latent_channels(vae)
    channels = int(sampler_latent_channels or vae_channels)
    slot_channels = int(concat_slot_channels or vae_channels)
    concat_slots = max(1, int(concat_slots))
    latent_t = latent_length(length)
    latent_h = height // scale
    latent_w = width // scale
    active_guides = [guide for guide in parse_guides_json(guides_json) if guide.enabled]
    if control_video is not None and concat_slots < 2 and (start_images is not None or active_guides):
        raise ValueError("This WAN model exposes only one image-conditioning slot, so control_video cannot be combined with start_images or manual guide frames.")
    latent = create_empty_latent(width, height, length, batch_size, vae, latent_channels=channels)
    device = latent["samples"].device
    dtype = latent["samples"].dtype

    timeline, mask_frames = build_guide_timeline(
        vae=vae,
        width=width,
        height=height,
        length=length,
        fps=fps,
        timing_mode=timing_mode,
        resize_mode=resize_mode,
        duplicate_policy=duplicate_policy,
        pad_color=pad_color,
        global_strength=global_strength,
        guides_json=guides_json,
        start_images=start_images,
        start_images_strength=start_images_strength,
    )
    concat_latent = create_wan22_concat_latent(batch_size, slot_channels, concat_slots, latent_t, latent_h, latent_w, device, dtype)
    guide_latent = vae.encode(timeline[:, :, :, :3])
    guide_latent = fit_latent_channels(guide_latent, slot_channels)
    guide_slot_start = slot_channels if concat_slots > 1 else 0
    concat_latent[:, guide_slot_start : guide_slot_start + slot_channels, : guide_latent.shape[2]] = guide_latent[
        :, :, : concat_latent.shape[2]
    ].to(device=device, dtype=dtype)

    if control_video is not None:
        control_video = upscale_images(control_video[:length], width, height)
        control_video = control_video.to(device=device, dtype=torch.float32)
        control_latent = vae.encode(control_video[:, :, :, :3])
        control_latent = fit_latent_channels(control_latent, slot_channels)
        concat_latent[:, :slot_channels, : control_latent.shape[2]] = control_latent[:, :, : concat_latent.shape[2]].to(
            device=device,
            dtype=dtype,
        )

    mask = mask_frames.view(1, latent_t, 4, latent_h, latent_w).transpose(1, 2).to(device=device, dtype=dtype)
    concat_mask_index = slot_channels if concat_slots > 1 else 0
    positive = node_helpers.conditioning_set_values(
        positive,
        {"concat_latent_image": concat_latent, "concat_mask": mask, "concat_mask_index": concat_mask_index},
    )
    negative = node_helpers.conditioning_set_values(
        negative,
        {"concat_latent_image": concat_latent, "concat_mask": mask, "concat_mask_index": concat_mask_index},
    )

    if ref_image is not None:
        ref_image = upscale_images(ref_image[:1], width, height)
        ref_latent = vae.encode(ref_image[:, :, :, :3])
        positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [ref_latent]}, append=True)
        negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [ref_latent]}, append=True)

    return positive, negative, latent
