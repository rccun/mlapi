from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
from .tasks import predict_biome
from .config import celery_app
from celery.result import AsyncResult



IMG_HEIGHT = 128
IMG_WIDTH = 128

app = FastAPI(title="Minecraft Biome Classifier")

@app.post("/predict/")
async def predict_biome(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        task = predict_biome.delay(contents)
        return {
            "task_id": task.id,
            "status": "processing"
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status/{task_id}")
def status(task_id: str):

    task = AsyncResult(task_id, app=celery_app)

    if task.state == "PENDING":
        return {"status": "pending"}

    if task.state == "SUCCESS":
        return {
            "status": "done",
            "result": task.result
        }

    if task.state == "FAILURE":
        return {"status": "error"}

    return {"status": task.state}

@app.delete("/task/{task_id}")
def delete(task_id: str):

    celery_app.control.revoke(task_id, terminate=True)

    return {"status": "deleted"}