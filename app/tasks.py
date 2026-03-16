from app.config import celery_app
from app.model import prepare_image, MODEL_PATH
from tensorflow.keras.models import load_model
from model.test import folders
import numpy as np

model = load_model(MODEL_PATH)

@celery_app.task(bind=True, name="app.tasks.predict_biome_task")
def predict_biome_task(self, file_bytes):

    img = prepare_image(file_bytes)

    preds = model.predict(img)


    class_idx = int(np.argmax(preds[0]))
    prob = float(preds[0][class_idx])

    biome = folders[class_idx]

    return {
        "biome": biome,
        "probability": prob
    }