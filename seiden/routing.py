from django.urls import path, re_path
from . import consumers

websocket_urlpatterns = [
    path('ws/progress/', consumers.ProgressConsumer.as_asgi()),
    path('ws/algorithm/', consumers.AlgorithmTypeConsumer.as_asgi()),
    # re_path('ws/progress/', consumers.ProgressConsumer.as_asgi()),
]


