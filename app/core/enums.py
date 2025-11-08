from enum import Enum


class QueueTypes(str, Enum):
    inference_queue = "inference_queue"
    inference_queue_result = "inference_queue_result"


class ModelStatus(str, Enum):
    pending = "pending"
    training = "training"
    completed = "completed"
    failed = "failed"