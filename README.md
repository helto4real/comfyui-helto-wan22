# comfyui-helto-wan22

Custom ComfyUI nodes for WAN 2.2 image-to-video guide workflows.

This package adds LTX-style multi-image guide management for WAN 2.2 while using WAN-native concat conditioning. Guide images are placed into a full neutral frame timeline, VAE-encoded once, and attached through `concat_latent_image`, `concat_mask`, and `concat_mask_index`.

## Nodes

- `WAN22ImageGuideManager`: builds reusable guide payloads.
- `WAN22ApplyImageGuides`: applies guide payloads to WAN conditioning and latent setup.
- `WAN22MultiImageI2VGuide`: all-in-one guide picker plus conditioning output.
- `WAN22GenerateAllInOne`: prompt encode, WAN guide conditioning, two-phase sampling, and VAE decode.

## Features

- Manual image guide insertion by frame or seconds.
- Start image sequences, optional reference image, and optional control video.
- Saved guide sets, folder browsing, thumbnails, and workflow-safe `guides_json`.
- Duplicate handling, resize modes, global strength, start image strength, shift control, and structural repulsion boost.
- WAN 2.2 high/low model sampling support.

## Installation

Clone or symlink this repository into your ComfyUI `custom_nodes` directory:

```bash
cd ~/git/ComfyUI/custom_nodes
ln -s ~/git/comfyui-helto-wan22 comfyui-helto-wan22
```

Restart ComfyUI after installation.

## License

MIT
