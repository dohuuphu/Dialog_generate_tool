from django.urls import include, path, re_path
from modules.transcript import Transcript
from modules.normalize import Normalize

urlpatterns = [
    path('transcript/', Transcript.as_view(), name='transcript'),
    path('normalize/', Normalize.as_view(), name='normalize'),
]

