from __future__ import annotations

import base64

from celery.result import AsyncResult
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from .config import celery_app
from .tasks import predict_biome_task

import os

app = FastAPI(title="Minecraft Biome Classifier")


@app.post("/predict/")
async def predict_biome_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        task = predict_biome_task.delay(base64.b64encode(contents).decode("utf-8"))
        return {
            "task_id": task.id,
            "status": "processing",
        }
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=500)


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
            "result": task.result,
        }
    if task.state == "FAILURE":
        return {
            "status": "error",
            "error": str(task.info),
        }

    return {"status": task.state}


@app.delete("/task/{task_id}")
def delete(task_id: str):
    celery_app.control.revoke(task_id, terminate=True)
    return {"status": "deleted"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
