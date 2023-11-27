from django.urls import path
from .views import user_like_artist_view, song2vec_view, tag_song2vec_view, song_info

urlpatterns = [
    path('user_like_artist/', user_like_artist_view.as_view()),
    path('song2vec/', song2vec_view.as_view()),
    path('tag_song2vec/', tag_song2vec_view.as_view()),
    path('music/song_info/', song_info, name='get_song_info'),
]