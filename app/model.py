from PIL import Image
import numpy as np
import io

MODEL_PATH = "model/minecraft_biome_model.keras"
def prepare_image(file_bytes):

    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize((128, 128))

    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    return img