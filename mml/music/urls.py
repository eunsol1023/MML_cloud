from django.urls import path
from .views import user_like_artist_view, song2vec_view, tag_song2vec_view, song_info, tag_song2vec_his

urlpatterns = [
    path('user_like_artist/', user_like_artist_view.as_view()),
    path('song2vec/', song2vec_view.as_view()),
    path('tag_song2vec/', tag_song2vec_view.as_view()),
    path('song_info/', song_info),
    path('tag_song2vec_his/', tag_song2vec_his)
]