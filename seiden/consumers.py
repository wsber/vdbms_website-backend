import json
from datetime import time

from channels.generic.websocket import WebsocketConsumer, AsyncWebsocketConsumer
from asgiref.sync import async_to_sync


class ProgressConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.group_name = "progress_group"
        await self.channel_layer.group_add(
            self.group_name,
            self.channel_name
        )
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            self.group_name,
            self.channel_name
        )

    async def receive(self, text_data=None, bytes_data=None):
        data = json.loads(text_data)
        progress = data['progress']
        algorithm_type = data.get('algorithm_type')
        await self.channel_layer.group_send(
            self.group_name,
            {
                'type': 'progress.update',
                'progress': progress,
                'algorithm_type': algorithm_type
            }
        )

    async def progress_update(self, event):
        progress = event['progress']
        algorithm_type = event['algorithm_type']
        await self.send(text_data=json.dumps({
            'progress': progress,
            'algorithm_type': algorithm_type
        }))


class AlgorithmTypeConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.group_name = "algorithm_type_group"
        await self.channel_layer.group_add(
            self.group_name,
            self.channel_name
        )
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            self.group_name,
            self.channel_name
        )

    async def receive(self, text_data=None, bytes_data=None):
        data = json.loads(text_data)
        algorithm_type = data['algorithm_type']
        consumer_type = data.get('consumer_type')
        await self.channel_layer.group_send(
            self.group_name,
            {
                'type': 'progress.update',
                'progress': algorithm_type,
                'consumer_type': consumer_type
            }
        )

    async def algorithm_update(self, event):
        algorithm_type = event['algorithm_type']
        consumer_type = event['consumer_type']
        await self.send(text_data=json.dumps({
            'algorithm_type': algorithm_type,
            'consumer_type': consumer_type
        }))
