import json
import pika
from app.config import settings

# Его существование находится под вопросом, он мне не нужен скорее всего но пусть будет

#TODO: Добавить логгер и убрать принты
class RabbitMQPublisher:
    def __init__(self):
        self._connection = pika.BlockingConnection(
            pika.URLParameters(settings.RABBITMQ_URL)
        )
        self._channel = self._connection.channel()

    def publish(self, queue_name: str, message: dict):
        self._channel.queue_declare(queue=queue_name, durable=True)
        body = json.dumps(message, default=str)
        self._channel.basic_publish(
            exchange="",
            routing_key=queue_name,
            body=body,
            properties=pika.BasicProperties(delivery_mode=2),
        )

    def close(self):
        try:
            if self._connection and self._connection.is_open:
                self._connection.close()
        except Exception as e:
            print(f"Error closing RabbitMQ connection: {e}")

    def __del__(self):
        self.close()
