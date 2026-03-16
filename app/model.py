from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
import keras

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATHS = [
    # MODEL_DIR / "minecraft_biome_model.h5",
    MODEL_DIR / "biome_model.keras",
]
IMG_SIZE = (128, 128)


def prepare_image(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


@lru_cache(maxsize=1)
def get_model():
    errors: list[str] = []
    for model_path in MODEL_PATHS:
        if not model_path.exists():
            errors.append(f"not found: {model_path}")
            continue
        try:
            return keras.models.load_model(model_path, compile=False)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{model_path.name}: {exc}")

    joined = " | ".join(errors) if errors else "no model files found"
    raise RuntimeError(f"Could not load model. {joined}")
