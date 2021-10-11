from django.urls import path

from .views import (
    VideosView
)

app_name = 'videos'

urlpatterns = [
    path('search-video/<str:search_content>', VideosView, name='search')
]
