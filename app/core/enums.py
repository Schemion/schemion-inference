from enum import Enum


class QueueTypes(str, Enum):
    inference_queue = "inference_queue"
    inference_queue_result = "inference_queue_result"


class TaskStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"