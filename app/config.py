import os
from celery import Celery

broker = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
backend = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0")

celery_app = Celery(
    "mlapi",
    broker=broker,
    backend=backend,
    include=["app.tasks"],
)

celery_app.conf.task_routes = {
    "app.tasks.predict_biome_task": {"queue": "ml_queue"},
}
celery_app.conf.task_serializer = "json"
celery_app.conf.result_serializer = "json"
celery_app.conf.accept_content = ["json"]
celery_app.conf.result_expires = 3600
