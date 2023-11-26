from django.urls import path
from .views import FaceDatasetView


app_name = 'core'

urlpatterns = [
    path('', FaceDatasetView.as_view(), name='face_dataset'),
]