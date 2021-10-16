from django.urls import path
from videos import views


app_name = 'videos'

urlpatterns = [
    path('search-video/', views.VideosView, name='search')
]