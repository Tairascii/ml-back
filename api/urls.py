from django.urls import path

from .views import upload_image_and_process

urlpatterns = [
    path('process-image/', upload_image_and_process, name='process_video'),
]