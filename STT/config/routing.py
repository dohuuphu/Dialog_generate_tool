from django.urls import path

from . import consumers

ws_urlpatterns = [
    path('ws/realtime/<str:room_name>/', consumers.APIConsumer.as_asgi()),
]