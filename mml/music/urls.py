from django.urls import path
from .views import music_reco_view, user_like_artist_view, song2vec_view, tag_song2vec_view

urlpatterns = [
    path('music_reco/', music_reco_view.as_view()),
    path('user_like_artist/', user_like_artist_view.as_view()),
    path('song2vec/', song2vec_view.as_view()),
    path('tag_song2vec/', tag_song2vec_view.as_view()),
]