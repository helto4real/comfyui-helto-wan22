import json

from .guide_models import guide_summary, parse_guides_json
from .wan22_guides import apply_wan22_guides

MAX_RESOLUTION = 16384
DEFAULT_GUIDES_JSON = "{\"version\":1,\"guides\":[]}"
GUIDE_DEFAULTS = {
    "fps": 24.0,
    "length": 49,
    "timing_mode": "frame",
    "resize_mode": "contain",
    "duplicate_policy": "error",
    "pad_color": "0,0,0",
    "global_strength": 1.0,
    "start_images_strength": 0.85,
}
GENERATION_DEFAULTS = {
    "seed": 0,
    "steps": 30,
    "cfg": 1.0,
    "sampler_name": "euler",
    "scheduler": "simple",
    "denoise": 1.0,
    "switch_step": 3,
}


def sampler_names():
    try:
        import comfy.samplers

        return list(getattr(comfy.samplers.KSampler, "SAMPLERS", ["euler"]))
    except Exception:
        return ["euler"]


def scheduler_names():
    try:
        import comfy.samplers

        return list(getattr(comfy.samplers.KSampler, "SCHEDULERS", ["simple"]))
    except Exception:
        return ["simple"]


def default_sampler_name():
    names = sampler_names()
    preferred = GENERATION_DEFAULTS["sampler_name"]
    return preferred if preferred in names else names[0]


def default_scheduler_name():
    names = scheduler_names()
    preferred = GENERATION_DEFAULTS["scheduler"]
    return preferred if preferred in names else names[0]


def coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def parse_guides_json_payload(guides_json):
    if isinstance(guides_json, dict):
        return guides_json
    return json.loads(guides_json or DEFAULT_GUIDES_JSON)


def guides_setting(guides_json, key):
    try:
        payload = parse_guides_json_payload(guides_json)
    except Exception:
        payload = {}
    return payload.get(key, GUIDE_DEFAULTS[key])


def build_guides_payload(guides_json, **settings):
    try:
        payload = parse_guides_json_payload(guides_json)
    except Exception:
        payload = {"version": 1, "guides": []}
    payload = dict(payload)
    payload["version"] = payload.get("version", 1)
    payload["guides"] = payload.get("guides", [])
    for key, value in settings.items():
        payload[key] = value
    return json.dumps(payload)


def encode_prompt(clip, text):
    if clip is None:
        raise RuntimeError("CLIP input is required to encode prompts.")
    tokens = clip.tokenize(text or "")
    return clip.encode_from_tokens_scheduled(tokens)


def decode_video_latent(vae, latent):
    samples = latent["samples"]
    if getattr(samples, "is_nested", False):
        samples = samples.unbind()[0]
    images = vae.decode(samples)
    if len(images.shape) == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images


def model_latent_channels(model):
    try:
        latent_format = model.get_model_object("latent_format")
        return int(latent_format.latent_channels)
    except Exception:
        return None


def model_concat_slots(model, latent_channels):
    try:
        diffusion_model = model.get_model_object("diffusion_model")
        patch_channels = int(diffusion_model.patch_embedding.weight.shape[1])
        extra_channels = patch_channels - int(latent_channels)
        if extra_channels <= 0:
            return 0
        image_channels = extra_channels - 4
        if image_channels <= 0:
            return 0
        return max(1, image_channels // int(latent_channels))
    except Exception:
        return 2


def sample_wan22_video(
    high_model,
    low_model,
    positive_high,
    positive_low,
    negative,
    latent,
    seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    denoise,
    switch_step,
):
    import nodes as comfy_nodes

    steps = int(steps)
    switch_step = max(0, min(steps, int(switch_step)))
    seed = int(seed)
    cfg = float(cfg)
    denoise = float(denoise)

    if switch_step > 0:
        latent = comfy_nodes.common_ksampler(
            model=high_model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive_high,
            negative=negative,
            latent=latent,
            denoise=denoise,
            start_step=0,
            last_step=switch_step,
            force_full_denoise=False,
        )[0]

    if switch_step < steps:
        latent = comfy_nodes.common_ksampler(
            model=low_model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive_low,
            negative=negative,
            latent=latent,
            denoise=denoise,
            disable_noise=switch_step > 0,
            start_step=switch_step,
            last_step=10000,
            force_full_denoise=True,
        )[0]

    return latent


def guide_settings_input_types(include_start_strength=True):
    settings = {
        "fps": ("FLOAT", {"default": GUIDE_DEFAULTS["fps"], "min": 1.0, "max": 240.0, "step": 0.01, "tooltip": "Frames per second used when timing_mode is seconds."}),
        "timing_mode": (["frame", "seconds"], {"default": GUIDE_DEFAULTS["timing_mode"], "tooltip": "Interpret guide positions as frame indexes or seconds."}),
        "resize_mode": (["contain", "pad", "stretch", "crop"], {"default": GUIDE_DEFAULTS["resize_mode"], "tooltip": "How guide images are resized before VAE encoding. contain/pad preserves aspect ratio with padding."}),
        "duplicate_policy": (["error", "keep_first", "keep_last", "offset_next"], {"default": GUIDE_DEFAULTS["duplicate_policy"], "tooltip": "How to handle guide images that resolve to the same frame."}),
        "pad_color": ("STRING", {"default": GUIDE_DEFAULTS["pad_color"], "tooltip": "RGB padding color for contain/pad resize mode. Accepts r,g,b or #rrggbb."}),
        "global_strength": ("FLOAT", {"default": GUIDE_DEFAULTS["global_strength"], "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Multiplier applied to every guide strength and start image strength."}),
    }
    if include_start_strength:
        settings["start_images_strength"] = ("FLOAT", {"default": GUIDE_DEFAULTS["start_images_strength"], "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Strength for the optional start image sequence before global_strength is applied."})
    return settings


def stage_input_types(include_guides_json=False, include_image_guides=False, include_settings=True):
    required = {
        "positive": ("CONDITIONING", {"tooltip": "Positive conditioning to augment with WAN 2.2 image guide metadata."}),
        "negative": ("CONDITIONING", {"tooltip": "Negative conditioning to augment with matching WAN 2.2 image guide metadata."}),
        "vae": ("VAE", {"tooltip": "WAN 2.2 VAE used to encode inserted guide frames."}),
        "width": ("INT", {"default": 1280, "min": 32, "max": MAX_RESOLUTION, "step": 32}),
        "height": ("INT", {"default": 704, "min": 32, "max": MAX_RESOLUTION, "step": 32}),
        "length": ("INT", {"default": GUIDE_DEFAULTS["length"], "min": 1, "max": MAX_RESOLUTION, "step": 4, "tooltip": "Output video frame count. WAN latent length is ((length - 1) // 4) + 1."}),
        "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
    }
    if include_settings:
        required.update(guide_settings_input_types(include_start_strength=True))
    if include_guides_json:
        required["guides_json"] = ("STRING", {"default": DEFAULT_GUIDES_JSON, "tooltip": "Hidden serialized guide data used by the custom UI and saved in workflows."})
    if include_image_guides:
        required["image_guides"] = ("WAN22_IMAGE_GUIDES", {"tooltip": "Reusable guide payload from WAN 2.2 Image Guide Manager."})
    return {
        "required": required,
        "optional": {
            "start_images": ("IMAGE", {"tooltip": "Optional IMAGE batch inserted from frame 0."}),
            "ref_image": ("IMAGE", {"tooltip": "Optional WAN 2.2 reference image encoded as reference_latents."}),
            "control_video": ("IMAGE", {"tooltip": "Optional WAN 2.2 control video encoded into the first concat slot."}),
        },
    }


def generation_input_types():
    return {
        "required": {
            "high_model": ("MODEL", {"tooltip": "WAN 2.2 high-noise image-to-video model, usually sampled first."}),
            "low_model": ("MODEL", {"tooltip": "WAN 2.2 low-noise image-to-video model, usually sampled after switch_step."}),
            "clip": ("CLIP", {"tooltip": "WAN text encoder used to encode prompts."}),
            "vae": ("VAE", {"tooltip": "WAN 2.2 VAE used for guide encoding and final decode."}),
            "positive_prompt": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True}),
            "negative_prompt": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True}),
            "width": ("INT", {"default": 1280, "min": 32, "max": MAX_RESOLUTION, "step": 32}),
            "height": ("INT", {"default": 704, "min": 32, "max": MAX_RESOLUTION, "step": 32}),
            "length": ("INT", {"default": GUIDE_DEFAULTS["length"], "min": 1, "max": MAX_RESOLUTION, "step": 4}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            **guide_settings_input_types(include_start_strength=True),
            "seed": ("INT", {"default": GENERATION_DEFAULTS["seed"], "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            "steps": ("INT", {"default": GENERATION_DEFAULTS["steps"], "min": 1, "max": 10000}),
            "cfg": ("FLOAT", {"default": GENERATION_DEFAULTS["cfg"], "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
            "sampler_name": (sampler_names(), {"default": default_sampler_name()}),
            "scheduler": (scheduler_names(), {"default": default_scheduler_name()}),
            "switch_step": ("INT", {"default": GENERATION_DEFAULTS["switch_step"], "min": 0, "max": 10000, "tooltip": "Step where sampling switches from high_model to low_model. Common WAN 2.2 workflows use 3 with 5 total steps."}),
            "denoise": ("FLOAT", {"default": GENERATION_DEFAULTS["denoise"], "min": 0.0, "max": 1.0, "step": 0.01}),
            "guides_json": ("STRING", {"default": DEFAULT_GUIDES_JSON, "tooltip": "Hidden serialized guide data used by the custom UI and saved in workflows."}),
        },
        "optional": {
            "start_images": ("IMAGE", {"tooltip": "Optional IMAGE batch inserted from frame 0."}),
            "ref_image": ("IMAGE", {"tooltip": "Optional WAN 2.2 reference image encoded as reference_latents."}),
            "control_video": ("IMAGE", {"tooltip": "Optional WAN 2.2 control video encoded into the first concat slot."}),
        },
    }


def run_apply_guides(
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
    if resize_mode == "pad":
        resize_mode = "contain"
    return apply_wan22_guides(
        positive=positive,
        negative=negative,
        vae=vae,
        width=width,
        height=height,
        length=length,
        batch_size=batch_size,
        fps=fps,
        timing_mode=timing_mode,
        resize_mode=resize_mode,
        duplicate_policy=duplicate_policy,
        pad_color=pad_color,
        global_strength=global_strength,
        guides_json=guides_json,
        start_images=start_images,
        start_images_strength=start_images_strength,
        ref_image=ref_image,
        control_video=control_video,
        sampler_latent_channels=sampler_latent_channels,
        concat_slots=concat_slots,
        concat_slot_channels=concat_slot_channels,
    )


class WAN22MultiImageI2VGuide:
    @classmethod
    def INPUT_TYPES(cls):
        return stage_input_types(include_guides_json=True)

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive_high", "positive_low", "negative", "latent")
    FUNCTION = "run"
    CATEGORY = "WAN 2.2"
    DESCRIPTION = "All-in-one WAN 2.2 image guide node: select images, insert them into the frame timeline, and output guided conditioning plus latent."

    @classmethod
    def IS_CHANGED(cls, guides_json, **kwargs):
        try:
            return guide_summary(parse_guides_json(guides_json))
        except Exception:
            return guides_json

    def run(self, positive, negative, vae, width, height, length, batch_size, fps, timing_mode, resize_mode, duplicate_policy, pad_color, global_strength, start_images_strength, guides_json, start_images=None, ref_image=None, control_video=None):
        positive_guided, negative_guided, latent = run_apply_guides(positive, negative, vae, width, height, length, batch_size, fps, timing_mode, resize_mode, duplicate_policy, pad_color, global_strength, guides_json, start_images, start_images_strength, ref_image, control_video)
        return positive_guided, positive_guided, negative_guided, latent


class WAN22ImageGuideManager:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **guide_settings_input_types(include_start_strength=True),
                "width": ("INT", {"default": 1280, "min": 32, "max": MAX_RESOLUTION, "step": 32, "tooltip": "Preview target width used for aspect-ratio warnings."}),
                "height": ("INT", {"default": 704, "min": 32, "max": MAX_RESOLUTION, "step": 32, "tooltip": "Preview target height used for aspect-ratio warnings."}),
                "length": ("INT", {"default": GUIDE_DEFAULTS["length"], "min": 1, "max": MAX_RESOLUTION, "step": 4, "tooltip": "Output video frame count used for guide timing."}),
                "guides_json": ("STRING", {"default": DEFAULT_GUIDES_JSON, "tooltip": "Hidden serialized guide data used by the custom UI and saved in workflows."}),
            }
        }

    RETURN_TYPES = ("WAN22_IMAGE_GUIDES",)
    RETURN_NAMES = ("image_guides",)
    FUNCTION = "run"
    CATEGORY = "WAN 2.2"
    DESCRIPTION = "Build a reusable WAN 2.2 image guide set and shared timing/strength settings."

    @classmethod
    def IS_CHANGED(cls, guides_json, **kwargs):
        try:
            return guide_summary(parse_guides_json(guides_json))
        except Exception:
            return guides_json

    def run(self, fps, timing_mode, resize_mode, duplicate_policy, pad_color, global_strength, start_images_strength, width, height, length, guides_json):
        return (
            build_guides_payload(
                guides_json,
                fps=float(fps),
                timing_mode=timing_mode,
                resize_mode=resize_mode,
                duplicate_policy=duplicate_policy,
                pad_color=pad_color,
                global_strength=float(global_strength),
                start_images_strength=float(start_images_strength),
                width=int(width),
                height=int(height),
                length=int(length),
            ),
        )


class WAN22ApplyImageGuides:
    @classmethod
    def INPUT_TYPES(cls):
        return stage_input_types(include_image_guides=True, include_settings=False)

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive_high", "positive_low", "negative", "latent")
    FUNCTION = "run"
    CATEGORY = "WAN 2.2"
    DESCRIPTION = "Apply a WAN22_IMAGE_GUIDES payload to one WAN 2.2 image-to-video sampler stage."

    @classmethod
    def IS_CHANGED(cls, image_guides, **kwargs):
        try:
            return guide_summary(parse_guides_json(image_guides))
        except Exception:
            return image_guides

    def run(self, positive, negative, vae, width, height, length, batch_size, image_guides, start_images=None, ref_image=None, control_video=None, **_legacy_inputs):
        positive_guided, negative_guided, latent = run_apply_guides(
            positive,
            negative,
            vae,
            width,
            height,
            length,
            batch_size,
            guides_setting(image_guides, "fps"),
            guides_setting(image_guides, "timing_mode"),
            guides_setting(image_guides, "resize_mode"),
            guides_setting(image_guides, "duplicate_policy"),
            guides_setting(image_guides, "pad_color"),
            guides_setting(image_guides, "global_strength"),
            image_guides,
            start_images,
            guides_setting(image_guides, "start_images_strength"),
            ref_image,
            control_video,
        )
        return positive_guided, positive_guided, negative_guided, latent


class WAN22GenerateAllInOne:
    @classmethod
    def INPUT_TYPES(cls):
        return generation_input_types()

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "WAN 2.2"
    DESCRIPTION = "Single-pass WAN 2.2 generation with prompts, frame guide insertion, sampling, and VAE decode."

    @classmethod
    def IS_CHANGED(cls, guides_json=DEFAULT_GUIDES_JSON, **kwargs):
        try:
            guides_part = guide_summary(parse_guides_json(guides_json))
        except Exception:
            guides_part = guides_json
        tracked = [
            "positive_prompt",
            "negative_prompt",
            "width",
            "height",
            "length",
            "batch_size",
            "fps",
            "timing_mode",
            "resize_mode",
            "duplicate_policy",
            "pad_color",
            "global_strength",
            "start_images_strength",
            "seed",
            "steps",
            "cfg",
            "sampler_name",
            "scheduler",
            "switch_step",
            "denoise",
        ]
        return (guides_part, tuple((name, kwargs.get(name)) for name in tracked))

    def run(self, high_model, low_model, clip, vae, positive_prompt, negative_prompt, width, height, length, batch_size, fps, timing_mode, resize_mode, duplicate_policy, pad_color, global_strength, start_images_strength, seed, steps, cfg, sampler_name, scheduler, switch_step, denoise, guides_json, start_images=None, ref_image=None, control_video=None):
        if sampler_name not in sampler_names():
            raise ValueError(f"Unknown sampler_name: {sampler_name}")
        if scheduler not in scheduler_names():
            raise ValueError(f"Unknown scheduler: {scheduler}")
        high_channels = model_latent_channels(high_model)
        low_channels = model_latent_channels(low_model)
        uses_high = int(switch_step) > 0
        uses_low = int(switch_step) < int(steps)
        if uses_high and uses_low and high_channels and low_channels and high_channels != low_channels:
            raise ValueError(f"WAN high_model and low_model latent channel counts differ: {high_channels} vs {low_channels}.")
        sampler_channels = high_channels if uses_high else low_channels
        high_slots = model_concat_slots(high_model, high_channels) if uses_high and high_channels else 2
        low_slots = model_concat_slots(low_model, low_channels) if uses_low and low_channels else high_slots
        concat_slots = min(high_slots, low_slots) if uses_high and uses_low else (high_slots if uses_high else low_slots)
        if concat_slots <= 0:
            raise ValueError("The selected high_model does not appear to expose WAN image-conditioning channels.")
        positive = encode_prompt(clip, positive_prompt)
        negative = encode_prompt(clip, negative_prompt)
        positive_guided, negative_guided, latent = run_apply_guides(
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
            start_images,
            start_images_strength,
            ref_image,
            control_video,
            sampler_latent_channels=sampler_channels,
            concat_slots=concat_slots,
            concat_slot_channels=sampler_channels,
        )
        sampled = sample_wan22_video(
            high_model,
            low_model,
            positive_guided,
            positive_guided,
            negative_guided,
            latent,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            denoise,
            switch_step,
        )
        return (decode_video_latent(vae, sampled),)


NODE_CLASS_MAPPINGS = {
    "WAN22MultiImageI2VGuide": WAN22MultiImageI2VGuide,
    "WAN22ImageGuideManager": WAN22ImageGuideManager,
    "WAN22ApplyImageGuides": WAN22ApplyImageGuides,
    "WAN22GenerateAllInOne": WAN22GenerateAllInOne,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WAN22MultiImageI2VGuide": "WAN 2.2 Image Guides (All-in-One)",
    "WAN22ImageGuideManager": "WAN 2.2 Image Guide Manager",
    "WAN22ApplyImageGuides": "WAN 2.2 Apply Image Guides",
    "WAN22GenerateAllInOne": "WAN 2.2 Generate All-in-One",
}


try:
    from . import routes  # noqa: F401
except Exception as exc:
    print(f"[WAN 2.2 Image Guides] Failed to register routes: {exc}")
