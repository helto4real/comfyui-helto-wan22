import torch

from .config_store import resolve_image_path
from .guide_models import parse_guides_json
from .image_io import load_guide_tensor, resize_tensor_images


def clamp01(value):
    return max(0.0, min(1.0, float(value)))


def resolve_timing(guides, timing_mode, fps, length, duplicate_policy, duplicate_group_size=1):
    resolved = []
    occupied = {}
    length = max(1, int(length))
    duplicate_group_size = max(1, int(duplicate_group_size))
    for guide in guides:
        if not guide.enabled:
            continue
        frame = round(guide.position * fps) if timing_mode == "seconds" else int(round(guide.position))
        if frame < 0:
            frame = length + frame
        frame = max(0, min(length - 1, frame))
        key = frame // duplicate_group_size
        if key in occupied:
            if duplicate_policy == "error":
                raise ValueError(f"Duplicate guide latent group at frame {frame}: {guide.filename}")
            if duplicate_policy == "keep_first":
                continue
            if duplicate_policy == "keep_last":
                prior = occupied[key]
                resolved[prior] = None
                occupied[key] = len(resolved)
            if duplicate_policy == "offset_next":
                while key in occupied and frame < length - 1:
                    frame = min(length - 1, (key + 1) * duplicate_group_size)
                    key = frame // duplicate_group_size
                if key in occupied:
                    raise ValueError(f"Could not offset duplicate guide near final frame for {guide.filename}")
                occupied[key] = len(resolved)
        else:
            occupied[key] = len(resolved)
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


def validate_latent_channels(latent, channels, source):
    channels = int(channels)
    if latent.shape[1] == channels:
        return latent
    raise ValueError(
        f"{source} encoded to {latent.shape[1]} latent channels, but the selected WAN model expects {channels}. "
        "Use a matching WAN model/VAE pair for image guide conditioning; channels are no longer sliced or padded because that degrades image fidelity."
    )


def upscale_images(images, width, height):
    import comfy.utils

    return comfy.utils.common_upscale(images.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)


def encode_wan_images(vae, images, slot_channels, source):
    latent = vae.encode(images[:, :, :, :3])
    return validate_latent_channels(latent, slot_channels, source)


def write_latent_group(concat_latent, slot_start, slot_channels, latent, latent_index, device, dtype):
    latent_index = int(latent_index)
    if latent_index >= concat_latent.shape[2]:
        return
    write_t = min(latent.shape[2], concat_latent.shape[2] - latent_index)
    if write_t <= 0:
        return
    concat_latent[:, slot_start : slot_start + slot_channels, latent_index : latent_index + write_t] = latent[
        :, :, :write_t
    ].to(device=device, dtype=dtype)


def lock_pixel_frames(mask_frames, start_frame, end_frame, strength):
    start_frame = max(0, int(start_frame))
    end_frame = min(mask_frames.shape[2], int(end_frame))
    if start_frame >= end_frame:
        return
    mask_frames[:, :, start_frame:end_frame] = 1.0 - clamp01(strength)


def create_spatial_gradient(img1, img2, mask_h, mask_w, boost):
    import torch.nn.functional as F

    if boost <= 1.0:
        return None
    motion_diff = torch.abs(img2[0, :, :, :3] - img1[0, :, :, :3]).mean(dim=-1)
    motion_diff = motion_diff.unsqueeze(0).unsqueeze(0)
    motion_diff = F.interpolate(motion_diff, size=(mask_h, mask_w), mode="bilinear", align_corners=False)
    motion_min = motion_diff.min()
    motion_max = motion_diff.max()
    normalized = (motion_diff - motion_min) / (motion_max - motion_min + 1e-8)
    spatial_gradient = 1.0 - normalized * (float(boost) - 1.0) * 2.5
    return torch.clamp(spatial_gradient[0, 0], 0.02, 1.0)


def apply_structural_repulsion(mask_frames, anchors, boost):
    if float(boost) <= 1.0 or len(anchors) < 2:
        return mask_frames
    mask_h, mask_w = mask_frames.shape[-2], mask_frames.shape[-1]
    for (pos1, img1), (pos2, img2) in zip(anchors, anchors[1:]):
        pos1 = int(pos1)
        pos2 = int(pos2)
        if pos2 <= pos1 + 4:
            continue
        transition_start = pos1 + 4
        transition_end = min(pos2 - 4, pos2)
        if transition_start >= transition_end:
            continue
        spatial_gradient = create_spatial_gradient(img1, img2, mask_h, mask_w, boost)
        if spatial_gradient is None:
            continue
        for frame_idx in range(transition_start, transition_end):
            if frame_idx >= mask_frames.shape[2]:
                break
            mask_frames[:, :, frame_idx, :, :] = mask_frames[:, :, frame_idx, :, :] * spatial_gradient
    return mask_frames


def build_guide_conditioning(
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
    batch_size,
    slot_channels,
    concat_slots,
    latent_t,
    latent_h,
    latent_w,
    device,
    dtype,
    start_images=None,
    start_images_strength=0.85,
    control_video=None,
    structural_repulsion_boost=1.0,
):
    guides = parse_guides_json(guides_json)
    resolved = resolve_timing(guides, timing_mode, float(fps), length, duplicate_policy, duplicate_group_size=4)
    concat_latent = create_wan22_concat_latent(batch_size, slot_channels, concat_slots, latent_t, latent_h, latent_w, device, dtype)
    timeline = torch.ones((int(length), int(height), int(width), 3), device=device, dtype=torch.float32) * 0.5
    mask_low = torch.ones((1, 1, latent_t * 4, latent_h, latent_w), device=device, dtype=torch.float32)
    anchors = []
    guide_slot_start = slot_channels if concat_slots > 1 else 0

    if start_images is not None:
        if start_images.shape[0] <= 0:
            raise ValueError("Start image sequence is empty.")
        start = resize_tensor_images(start_images[: int(length)], width, height, resize_mode, pad_color)
        start = start.to(device=device, dtype=torch.float32)
        sequence_len = min(start.shape[0], int(length))
        for frame_idx, guide in resolved:
            if frame_idx < min(int(length), sequence_len + 3):
                raise ValueError(f"Manual guide {guide.filename} at frame {frame_idx} overlaps start_images.")
        timeline[:sequence_len] = start[:sequence_len]
        start_strength = clamp01(global_strength) * clamp01(start_images_strength)
        lock_pixel_frames(mask_low, 0, sequence_len + 3, start_strength)
        anchors.append((0, start[:1]))

    for frame_idx, guide in resolved:
        image_path = resolve_image_path(guide.folder_alias, guide.filename)
        image, _ = load_guide_tensor(image_path, width, height, resize_mode, pad_color)
        image = image.to(device=device, dtype=torch.float32)
        latent_index = min(latent_t - 1, int(frame_idx) // 4)
        timeline[int(frame_idx) : int(frame_idx) + 1] = image[:1]
        guide_strength = clamp01(global_strength) * clamp01(guide.strength)
        lock_pixel_frames(mask_low, latent_index * 4, (latent_index + 1) * 4, guide_strength)
        anchors.append((frame_idx, image))

    guide_latent = encode_wan_images(vae, timeline, slot_channels, "guide timeline")
    write_latent_group(concat_latent, guide_slot_start, slot_channels, guide_latent, 0, device, dtype)

    if control_video is not None:
        control_video = upscale_images(control_video[:length], width, height)
        control_video = control_video.to(device=device, dtype=torch.float32)
        control_latent = encode_wan_images(vae, control_video, slot_channels, "control_video")
        write_latent_group(concat_latent, 0, slot_channels, control_latent, 0, device, dtype)

    anchors = sorted(anchors, key=lambda item: item[0])
    mask_high = apply_structural_repulsion(mask_low.clone(), anchors, structural_repulsion_boost)
    return concat_latent, mask_high, mask_low


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
    structural_repulsion_boost=1.0,
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

    concat_latent, mask_high_frames, mask_low_frames = build_guide_conditioning(
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
        batch_size=batch_size,
        slot_channels=slot_channels,
        concat_slots=concat_slots,
        latent_t=latent_t,
        latent_h=latent_h,
        latent_w=latent_w,
        device=device,
        dtype=dtype,
        start_images=start_images,
        start_images_strength=start_images_strength,
        control_video=control_video,
        structural_repulsion_boost=structural_repulsion_boost,
    )

    mask_high = mask_high_frames.view(1, latent_t, 4, latent_h, latent_w).transpose(1, 2).to(device=device, dtype=dtype)
    mask_low = mask_low_frames.view(1, latent_t, 4, latent_h, latent_w).transpose(1, 2).to(device=device, dtype=dtype)
    concat_mask_index = slot_channels if concat_slots > 1 else 0
    positive_high = node_helpers.conditioning_set_values(
        positive,
        {"concat_latent_image": concat_latent, "concat_mask": mask_high, "concat_mask_index": concat_mask_index},
    )
    positive_low = node_helpers.conditioning_set_values(
        positive,
        {"concat_latent_image": concat_latent, "concat_mask": mask_low, "concat_mask_index": concat_mask_index},
    )
    negative = node_helpers.conditioning_set_values(
        negative,
        {"concat_latent_image": concat_latent, "concat_mask": mask_high, "concat_mask_index": concat_mask_index},
    )

    if ref_image is not None:
        ref_image = upscale_images(ref_image[:1], width, height)
        ref_latent = vae.encode(ref_image[:, :, :, :3])
        ref_latent = validate_latent_channels(ref_latent, slot_channels, "ref_image")
        positive_high = node_helpers.conditioning_set_values(positive_high, {"reference_latents": [ref_latent]}, append=True)
        positive_low = node_helpers.conditioning_set_values(positive_low, {"reference_latents": [ref_latent]}, append=True)
        negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [ref_latent]}, append=True)

    return positive_high, positive_low, negative, latent
