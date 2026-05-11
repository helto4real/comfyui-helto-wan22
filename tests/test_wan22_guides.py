import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


def load_package_with_stubs():
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_input_directory = lambda: "/tmp"
    sys.modules["folder_paths"] = folder_paths

    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    model_management = types.ModuleType("comfy.model_management")
    model_management.intermediate_device = lambda: "cpu"
    latent_formats = types.ModuleType("comfy.latent_formats")

    class LatentFormat:
        def process_out(self, latent):
            return latent

    latent_formats.Wan22 = LatentFormat
    latent_formats.Wan21 = LatentFormat
    utils = types.ModuleType("comfy.utils")
    utils.common_upscale = lambda x, width, height, method, crop: torch.nn.functional.interpolate(
        x,
        size=(height, width),
        mode="nearest",
    )
    comfy.model_management = model_management
    comfy.latent_formats = latent_formats
    comfy.utils = utils
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = model_management
    sys.modules["comfy.latent_formats"] = latent_formats
    sys.modules["comfy.utils"] = utils

    node_helpers = types.ModuleType("node_helpers")

    def conditioning_set_values(conditioning, values, append=False):
        output = []
        for item in conditioning:
            next_item = dict(item)
            for key, value in values.items():
                if append and key in next_item and isinstance(next_item[key], list) and isinstance(value, list):
                    next_item[key] = next_item[key] + value
                else:
                    next_item[key] = value
            output.append(next_item)
        return output

    node_helpers.conditioning_set_values = conditioning_set_values
    sys.modules["node_helpers"] = node_helpers

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("wan22testpkg", root / "__init__.py", submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["wan22testpkg"] = module
    spec.loader.exec_module(module)
    return module


class FakeVAE:
    latent_channels = 48

    def spacial_compression_encode(self):
        return 16

    def encode(self, images):
        frames, height, width, _ = images.shape
        return torch.zeros((1, 48, ((frames - 1) // 4) + 1, height // 16, width // 16), dtype=torch.float32)


class RecordingFakeVAE(FakeVAE):
    def __init__(self):
        self.encoded_images = []

    def encode(self, images):
        self.encoded_images.append(images.detach().cpu().clone())
        return super().encode(images)


class ValueFakeVAE(FakeVAE):
    def encode(self, images):
        frames, height, width, _ = images.shape
        latent_t = ((frames - 1) // 4) + 1
        value = float(images[:, :, :, :3].mean())
        return torch.full((1, 48, latent_t, height // 16, width // 16), value, dtype=torch.float32)


def test_resolve_timing_duplicate_policies():
    load_package_with_stubs()
    from wan22testpkg.guide_models import GuideItem
    from wan22testpkg.wan22_guides import resolve_timing

    guides = [
        GuideItem("input", "first.png", 2, 0),
        GuideItem("input", "second.png", 2, 0),
        GuideItem("input", "last.png", -1, 0),
    ]

    assert [guide.filename for _, guide in resolve_timing(guides, "frame", 24, 9, "keep_first")] == [
        "first.png",
        "last.png",
    ]
    assert [guide.filename for _, guide in resolve_timing(guides, "frame", 24, 9, "keep_last")] == [
        "second.png",
        "last.png",
    ]
    assert [frame for frame, _ in resolve_timing(guides, "frame", 24, 9, "offset_next")] == [2, 3, 8]


def test_resolve_timing_duplicate_policies_can_use_wan_latent_groups():
    load_package_with_stubs()
    from wan22testpkg.guide_models import GuideItem
    from wan22testpkg.wan22_guides import resolve_timing

    guides = [
        GuideItem("input", "first.png", 0, 0),
        GuideItem("input", "second.png", 3, 3),
    ]

    with pytest.raises(ValueError, match="Duplicate guide latent group"):
        resolve_timing(guides, "frame", 24, 12, "error", duplicate_group_size=4)

    assert [frame for frame, _ in resolve_timing(guides, "frame", 24, 12, "offset_next", duplicate_group_size=4)] == [
        0,
        4,
    ]


def test_apply_guides_builds_wan22_concat_latent_and_mask():
    load_package_with_stubs()
    from wan22testpkg.wan22_guides import apply_wan22_guides

    positive_high, positive_low, negative, latent = apply_wan22_guides(
        positive=[{}],
        negative=[{}],
        vae=FakeVAE(),
        width=128,
        height=64,
        length=9,
        batch_size=2,
        fps=24,
        timing_mode="frame",
        resize_mode="stretch",
        duplicate_policy="error",
        pad_color="0,0,0",
        global_strength=1.0,
        guides_json="{\"version\":1,\"guides\":[]}",
        start_images=torch.ones((1, 64, 128, 3)),
        start_images_strength=0.5,
    )

    cond = positive_low[0]
    assert latent["samples"].shape == (2, 48, 3, 4, 8)
    assert cond["concat_latent_image"].shape == (2, 96, 3, 4, 8)
    assert cond["concat_mask"].shape == (1, 4, 3, 4, 8)
    assert cond["concat_mask_index"] == 48
    assert float(cond["concat_mask"][0, 0, 0, 0, 0]) == 0.5
    assert torch.equal(positive_high[0]["concat_mask"], positive_low[0]["concat_mask"])
    assert torch.equal(negative[0]["concat_mask"], positive_high[0]["concat_mask"])


def test_manual_guides_encode_full_neutral_timeline_and_lock_first_group(monkeypatch):
    load_package_with_stubs()
    import wan22testpkg.wan22_guides as wan22_guides

    def fake_load_guide_tensor(path, width, height, resize_mode, pad_color):
        return torch.full((1, height, width, 3), 0.75, dtype=torch.float32), (width, height)

    monkeypatch.setattr(wan22_guides, "resolve_image_path", lambda folder_alias, filename: filename)
    monkeypatch.setattr(wan22_guides, "load_guide_tensor", fake_load_guide_tensor)

    vae = RecordingFakeVAE()
    guides_json = (
        "{\"version\":1,\"guides\":["
        "{\"folder_alias\":\"input\",\"filename\":\"first.png\",\"position\":0,\"calculated_frame\":0,\"strength\":1.0}"
        "]}"
    )
    _, positive_low, _, _ = wan22_guides.apply_wan22_guides(
        positive=[{}],
        negative=[{}],
        vae=vae,
        width=128,
        height=64,
        length=9,
        batch_size=1,
        fps=24,
        timing_mode="frame",
        resize_mode="stretch",
        duplicate_policy="error",
        pad_color="0,0,0",
        global_strength=1.0,
        guides_json=guides_json,
    )

    assert len(vae.encoded_images) == 1
    timeline = vae.encoded_images[0]
    assert timeline.shape == (9, 64, 128, 3)
    assert torch.allclose(timeline[0], torch.full((64, 128, 3), 0.75))
    assert torch.allclose(timeline[1], torch.full((64, 128, 3), 0.5))
    assert torch.allclose(positive_low[0]["concat_mask"][0, :, 0], torch.zeros((4, 4, 8)))


def test_apply_guides_rejects_latent_channel_mismatch():
    load_package_with_stubs()
    from wan22testpkg.wan22_guides import apply_wan22_guides

    with pytest.raises(ValueError, match="encoded to 48 latent channels"):
        apply_wan22_guides(
            positive=[{}],
            negative=[{}],
            vae=FakeVAE(),
            width=128,
            height=64,
            length=9,
            batch_size=1,
            fps=24,
            timing_mode="frame",
            resize_mode="stretch",
            duplicate_policy="error",
            pad_color="0,0,0",
            global_strength=1.0,
            guides_json="{\"version\":1,\"guides\":[]}",
            start_images=torch.ones((1, 64, 128, 3)),
            start_images_strength=1.0,
            sampler_latent_channels=16,
            concat_slots=1,
            concat_slot_channels=16,
        )


def test_start_images_do_not_seed_sampler_latent_or_noise_mask():
    load_package_with_stubs()
    from wan22testpkg.wan22_guides import apply_wan22_guides

    positive_high, positive_low, negative, latent = apply_wan22_guides(
        positive=[{}],
        negative=[{}],
        vae=ValueFakeVAE(),
        width=128,
        height=64,
        length=9,
        batch_size=2,
        fps=24,
        timing_mode="frame",
        resize_mode="stretch",
        duplicate_policy="error",
        pad_color="0,0,0",
        global_strength=1.0,
        guides_json="{\"version\":1,\"guides\":[]}",
        start_images=torch.ones((1, 64, 128, 3)),
        start_images_strength=0.75,
    )

    assert torch.allclose(latent["samples"], torch.zeros_like(latent["samples"]))
    assert "noise_mask" not in latent
    assert torch.allclose(positive_low[0]["concat_mask"][0, 0, 0], torch.full((4, 8), 0.25))
    assert torch.equal(positive_high[0]["concat_mask"], positive_low[0]["concat_mask"])
    assert torch.equal(negative[0]["concat_mask"], positive_high[0]["concat_mask"])


def test_frame_zero_manual_guide_does_not_seed_sampler_latent(monkeypatch):
    load_package_with_stubs()
    import wan22testpkg.wan22_guides as wan22_guides

    def fake_load_guide_tensor(path, width, height, resize_mode, pad_color):
        return torch.full((1, height, width, 3), 0.8, dtype=torch.float32), (width, height)

    monkeypatch.setattr(wan22_guides, "resolve_image_path", lambda folder_alias, filename: filename)
    monkeypatch.setattr(wan22_guides, "load_guide_tensor", fake_load_guide_tensor)

    guides_json = (
        "{\"version\":1,\"guides\":["
        "{\"folder_alias\":\"input\",\"filename\":\"first.png\",\"position\":0,\"calculated_frame\":0,\"strength\":0.6}"
        "]}"
    )
    _, positive_low, _, latent = wan22_guides.apply_wan22_guides(
        positive=[{}],
        negative=[{}],
        vae=ValueFakeVAE(),
        width=128,
        height=64,
        length=9,
        batch_size=1,
        fps=24,
        timing_mode="frame",
        resize_mode="stretch",
        duplicate_policy="error",
        pad_color="0,0,0",
        global_strength=1.0,
        guides_json=guides_json,
    )

    assert torch.allclose(latent["samples"], torch.zeros_like(latent["samples"]))
    assert "noise_mask" not in latent
    assert torch.allclose(positive_low[0]["concat_mask"][0, :, 0], torch.full((4, 4, 8), 0.4))


def test_non_zero_manual_guide_does_not_seed_sampler_latent(monkeypatch):
    load_package_with_stubs()
    import wan22testpkg.wan22_guides as wan22_guides

    monkeypatch.setattr(wan22_guides, "resolve_image_path", lambda folder_alias, filename: filename)
    monkeypatch.setattr(
        wan22_guides,
        "load_guide_tensor",
        lambda path, width, height, resize_mode, pad_color: (torch.ones((1, height, width, 3)), (width, height)),
    )

    guides_json = (
        "{\"version\":1,\"guides\":["
        "{\"folder_alias\":\"input\",\"filename\":\"middle.png\",\"position\":8,\"calculated_frame\":8,\"strength\":1.0}"
        "]}"
    )
    _, _, _, latent = wan22_guides.apply_wan22_guides(
        positive=[{}],
        negative=[{}],
        vae=ValueFakeVAE(),
        width=128,
        height=64,
        length=13,
        batch_size=1,
        fps=24,
        timing_mode="frame",
        resize_mode="stretch",
        duplicate_policy="error",
        pad_color="0,0,0",
        global_strength=1.0,
        guides_json=guides_json,
    )

    assert torch.allclose(latent["samples"], torch.zeros_like(latent["samples"]))
    assert "noise_mask" not in latent


def test_structural_repulsion_only_changes_high_noise_transition_mask(monkeypatch):
    load_package_with_stubs()
    import wan22testpkg.wan22_guides as wan22_guides

    def fake_load_guide_tensor(path, width, height, resize_mode, pad_color):
        image = torch.zeros((1, height, width, 3), dtype=torch.float32)
        if "second" in str(path):
            image[:, :, width // 2 :, :] = 1.0
        return image, (width, height)

    monkeypatch.setattr(wan22_guides, "resolve_image_path", lambda folder_alias, filename: filename)
    monkeypatch.setattr(wan22_guides, "load_guide_tensor", fake_load_guide_tensor)

    guides_json = (
        "{\"version\":1,\"guides\":["
        "{\"folder_alias\":\"input\",\"filename\":\"first.png\",\"position\":0,\"calculated_frame\":0,\"strength\":1.0},"
        "{\"folder_alias\":\"input\",\"filename\":\"second.png\",\"position\":12,\"calculated_frame\":12,\"strength\":1.0}"
        "]}"
    )
    positive_high, positive_low, negative, _ = wan22_guides.apply_wan22_guides(
        positive=[{}],
        negative=[{}],
        vae=FakeVAE(),
        width=128,
        height=64,
        length=17,
        batch_size=1,
        fps=24,
        timing_mode="frame",
        resize_mode="stretch",
        duplicate_policy="error",
        pad_color="0,0,0",
        global_strength=1.0,
        guides_json=guides_json,
        structural_repulsion_boost=1.5,
    )

    high_mask = positive_high[0]["concat_mask"]
    low_mask = positive_low[0]["concat_mask"]
    transition_high = high_mask[0, 0, 1]
    transition_low = low_mask[0, 0, 1]

    assert torch.any(transition_high < transition_low)
    assert torch.allclose(transition_low, torch.ones_like(transition_low))
    assert torch.equal(negative[0]["concat_mask"], high_mask)
    assert float(high_mask.min()) >= 0.0
    assert float(high_mask.max()) <= 1.0


def test_structural_repulsion_default_leaves_high_and_low_masks_equal(monkeypatch):
    load_package_with_stubs()
    import wan22testpkg.wan22_guides as wan22_guides

    monkeypatch.setattr(wan22_guides, "resolve_image_path", lambda folder_alias, filename: filename)
    monkeypatch.setattr(
        wan22_guides,
        "load_guide_tensor",
        lambda path, width, height, resize_mode, pad_color: (torch.ones((1, height, width, 3)), (width, height)),
    )

    guides_json = (
        "{\"version\":1,\"guides\":["
        "{\"folder_alias\":\"input\",\"filename\":\"first.png\",\"position\":0,\"calculated_frame\":0,\"strength\":1.0},"
        "{\"folder_alias\":\"input\",\"filename\":\"second.png\",\"position\":12,\"calculated_frame\":12,\"strength\":1.0}"
        "]}"
    )
    positive_high, positive_low, _, _ = wan22_guides.apply_wan22_guides(
        positive=[{}],
        negative=[{}],
        vae=FakeVAE(),
        width=128,
        height=64,
        length=17,
        batch_size=1,
        fps=24,
        timing_mode="frame",
        resize_mode="stretch",
        duplicate_policy="error",
        pad_color="0,0,0",
        global_strength=1.0,
        guides_json=guides_json,
    )

    assert torch.equal(positive_high[0]["concat_mask"], positive_low[0]["concat_mask"])


def test_all_in_one_sampler_splits_high_and_low_models():
    load_package_with_stubs()
    from wan22testpkg.nodes import sample_wan22_video

    calls = []
    nodes = types.ModuleType("nodes")

    def common_ksampler(**kwargs):
        calls.append(kwargs)
        return ({"samples": torch.zeros((1, 48, 3, 4, 8)), "stage": kwargs["model"]},)

    nodes.common_ksampler = common_ksampler
    sys.modules["nodes"] = nodes

    out = sample_wan22_video(
        high_model="high",
        low_model="low",
        positive_high=[{"stage": "high"}],
        positive_low=[{"stage": "low"}],
        negative=[{}],
        latent={"samples": torch.zeros((1, 48, 3, 4, 8))},
        seed=123,
        steps=5,
        cfg=1.0,
        sampler_name="euler",
        scheduler="simple",
        denoise=1.0,
        switch_step=3,
    )

    assert out["stage"] == "low"
    assert [call["model"] for call in calls] == ["high", "low"]
    assert calls[0]["last_step"] == 3
    assert calls[0]["force_full_denoise"] is False
    assert calls[1]["start_step"] == 3
    assert calls[1]["disable_noise"] is True


def test_patch_model_sampling_shift_clones_model_and_sets_shift():
    load_package_with_stubs()
    from wan22testpkg.nodes import patch_model_sampling_shift

    class Sampling:
        multiplier = 1000

        def __init__(self):
            self.shift = None

        def set_parameters(self, shift=1.0, multiplier=1000):
            self.shift = shift
            self.multiplier = multiplier

    class Model:
        def __init__(self):
            self.sampling = Sampling()
            self.patched = None

        def clone(self):
            clone = Model()
            clone.sampling = self.sampling
            return clone

        def get_model_object(self, name):
            assert name == "model_sampling"
            return self.sampling

        def add_object_patch(self, name, value):
            assert name == "model_sampling"
            self.patched = value
            self.sampling = value

    model = Model()
    patched = patch_model_sampling_shift(model, 6.5)

    assert patched is not model
    assert patched.get_model_object("model_sampling").shift == 6.5
    assert model.get_model_object("model_sampling").shift is None
