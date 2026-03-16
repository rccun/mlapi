from __future__ import annotations

import base64

import numpy as np

from .config import celery_app
from .model import get_model, prepare_image

folders = [
    "badlands",
    "birch_forest",
    "cherry_grove",
    "dark_forest",
    "desert",
    "flower_forest",
    "forest",
    "frozen_peaks",
    "ice_spikes",
    "jungle",
    "mushroom_fields",
    "ocean",
    "pale_garden",
    "plains",
    "savanna",
    "snowy_plains",
    "stony_shore",
    "sunflower_plains",
    "taiga",
]


@celery_app.task(bind=True, name="app.tasks.predict_biome_task")
def predict_biome_task(self, file_b64: str):
    file_bytes = base64.b64decode(file_b64)
    img = prepare_image(file_bytes)
    model = get_model()
    preds = model.predict(img, verbose=0)

    class_idx = int(np.argmax(preds[0]))
    prob = float(preds[0][class_idx])
    biome = folders[class_idx]

    return {
        "biome": biome,
        "probability": round(prob, 6),
    }
