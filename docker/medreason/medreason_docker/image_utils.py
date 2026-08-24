from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image


def load_image(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")
    return Image.open(path).convert("RGB")


def load_images(paths: Iterable[Path]) -> list[Image.Image]:
    return [load_image(path) for path in paths]
