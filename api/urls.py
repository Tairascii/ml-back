from django.urls import path

from .views import upload_video_and_process, classify_letter

urlpatterns = [
    path('process-video/', upload_video_and_process, name='process_video'),
    path('classify-letter/', classify_letter, name='classify_letter'),
]