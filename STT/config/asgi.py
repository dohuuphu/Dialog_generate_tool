from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application
from config import routing
import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.local")


application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(
            routing.ws_urlpatterns
        )
    )
})
