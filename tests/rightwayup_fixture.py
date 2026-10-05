"""Offline pinned model fixtures and synthetic images; never read a production camera."""

import os
from pathlib import Path
import shutil
from PIL import Image, ImageDraw


def install_cache(storage, tiers=("fast",)):
    from blocs.rightwayup.assets import manifest, verified_bytes
    location = os.environ.get("RIGHTWAYUP_TEST_CACHE")
    assert location, "Prepare setup_model.py caches and set RIGHTWAYUP_TEST_CACHE to rightwayup-cache"
    source = Path(location)
    assert source.is_dir() and not source.is_symlink()
    target = Path(storage) / "rightwayup-cache"
    target.mkdir(parents=True, exist_ok=True)
    info = manifest()
    for tier in tiers:
        spec = info["assets"][info["tiers"][tier]["stem"]+"-int8.onnx"]
        assert verified_bytes(source/spec["filename"], spec)
        shutil.copy2(source/spec["filename"], target/spec["filename"])


def picture(root, name="frame.png"):
    path = Path(root)/name
    image = Image.new("RGB", (640, 480), (105, 174, 225))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 330, 640, 480), fill=(91, 135, 69))
    draw.rectangle((180, 200, 450, 365), fill=(184, 129, 93))
    draw.polygon([(165, 200), (315, 105), (465, 200)], fill=(76, 71, 66))
    draw.rectangle((289, 279, 342, 365), fill=(44, 54, 62))
    for x in (210, 380):
        draw.rectangle((x, 230, x+36, 267), fill=(203, 223, 242))
    image.save(path)
    return path
