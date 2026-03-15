from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np
import io

IMG_SIZE = (128,128)

MODEL_PATH = "model/minecraft_biome_model.h5"
model = load_model(MODEL_PATH)

def prepare_image(file_bytes):

    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)

    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    return img