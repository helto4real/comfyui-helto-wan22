import importlib.util
import sys
import types
from pathlib import Path

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


def test_apply_guides_builds_wan22_concat_latent_and_mask():
    load_package_with_stubs()
    from wan22testpkg.wan22_guides import apply_wan22_guides

    positive, _, latent = apply_wan22_guides(
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

    cond = positive[0]
    assert latent["samples"].shape == (2, 48, 3, 4, 8)
    assert cond["concat_latent_image"].shape == (2, 96, 3, 4, 8)
    assert cond["concat_mask"].shape == (1, 4, 3, 4, 8)
    assert cond["concat_mask_index"] == 48
    assert float(cond["concat_mask"][0, 0, 0, 0, 0]) == 0.5


def test_apply_guides_can_target_single_slot_wan_i2v_model():
    load_package_with_stubs()
    from wan22testpkg.wan22_guides import apply_wan22_guides

    positive, _, latent = apply_wan22_guides(
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

    cond = positive[0]
    assert latent["samples"].shape == (1, 16, 3, 4, 8)
    assert cond["concat_latent_image"].shape == (1, 16, 3, 4, 8)
    assert cond["concat_mask"].shape == (1, 4, 3, 4, 8)
    assert cond["concat_mask_index"] == 0
    assert float(cond["concat_mask"][0, 0, 0, 0, 0]) == 0.0


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
