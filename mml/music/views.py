from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .serializers import *

from .user_like_artist import user_like_artist_view

from .song2vec import song2vec_view

from .tag_song2vec import tag_song2vec_view

from .test_song2vec import test_song2vec_view