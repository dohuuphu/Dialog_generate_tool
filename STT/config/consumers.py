import json
from channels.generic.websocket import AsyncWebsocketConsumer
from time import sleep
from asr_model.realtime import RealtimeGenerator
from ubold.apps.apps import AppsConfig
from queue import Queue

class APIConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.bytes_data = Queue()
        print('connect')
        await self.accept()
        RealtimeGenerator(asr_model=AppsConfig.asr_model, normalizer=AppsConfig.normalizer, gector=AppsConfig.gector,bytes_data=self.bytes_data).start()
        room_name = self.scope['url_route']['kwargs']['room_name']
        await self.channel_layer.group_add(room_name, self.channel_name)
        print(f"Connected channel {self.channel_name}")
        # for i in range(200):
        #     print(i)
        #     self.send(json.dumps({'message': i}))
        #     sleep(1)

    async def receive(self, text_data='',bytes_data=None):
        # text_data_json = json.loads(text_data)
        # message = text_data_json['message']
        room_name = self.scope['url_route']['kwargs']['room_name']
        # print('Received:',room_name,bytes_data)
        # await self.send_channel('oke nha',room_name)
        if bytes_data:
            self.bytes_data.put(bytes_data)
        
    
    def send_channel(self, message, room_name):
        self.channel_layer.group_send(
            room_name,
            {
                'type': 'send_message',
                "message": message
            }
        )
        print('Sent:',room_name,message)

    async def send_message(self, res):
        await self.send(text_data=json.dumps({
            'response': res['message'],
        }))
    async def disconnect(self, close_code):
        room_name = self.scope['url_route']['kwargs']['room_name']
        await self.channel_layer.group_discard(room_name, self.channel_name)
        print(f"Disconnected channel {self.channel_name}")