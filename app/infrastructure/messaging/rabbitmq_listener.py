import json
import pika
from app.config import settings


#TODO: Добавить логгер и убрать принты
class RabbitMQListener:
    def __init__(self, queue_name: str, callback):
        self.queue_name = queue_name
        self.callback = callback
        self._connection = pika.BlockingConnection(
            pika.URLParameters(settings.RABBITMQ_URL)
        )
        self._channel = self._connection.channel()
        self._channel.queue_declare(queue=queue_name, durable=True)

    def start(self):
        def _callback(ch, method, _properties, body):
            try:
                message = json.loads(body)
                self.callback(message)
                ch.basic_ack(delivery_tag=method.delivery_tag)
            except Exception as e:
                print(f"Error processing message: {e}")
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

        self._channel.basic_consume(
            queue=self.queue_name, on_message_callback=_callback, auto_ack=False
        )

        try:
            self._channel.start_consuming()
        except KeyboardInterrupt:
            print("Stopping listener...")
        finally:
            self._connection.close()
