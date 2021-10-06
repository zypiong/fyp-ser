from django.urls import path

from .views import (
    videos
)

app_name = 'videos'

urlpatterns = [
    path('search-video/', videos.as_view(), name='search')
]