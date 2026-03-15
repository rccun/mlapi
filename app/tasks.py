from .config import celery_app
from app.model import prepare_image, model
from model.test import folders
import numpy as np

@celery_app.task(bind=True)
def predict_biome(self, file_bytes):

    img = prepare_image(file_bytes)

    preds = model.predict(img)

    class_idx = int(np.argmax(preds[0]))
    prob = float(preds[0][class_idx])

    biome = folders[str(class_idx)]

    return {
        "biome": biome,
        "probability": prob
    }