from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from user.models import MMLMusicInfo

from .serializers import *

from .user_like_artist import user_like_artist_view

from .song2vec import song2vec_view

from .tag_song2vec import tag_song2vec_view

import logging


# Create or get a logger
logger = logging.getLogger(__name__)

def song_info(request):
    # Check if it's a GET request
    if request.method == 'GET':
        # Get parameters from the request
        title = request.GET.get('title', None)
        artist = request.GET.get('artist', None)

        # Logging the received parameters for debugging
        logger.info(f"Received title: {title}, artist: {artist}")

        # Validate the parameters
        if not title or not artist:
            return JsonResponse({'error': 'Missing required parameters'}, status=400)

        try:
            # Query the MMLMusicInfo model
            song_info = MMLMusicInfo.objects.filter(title=title, artist=artist).first()

            # Check if song exists
            if song_info:
                # Prepare the response data
                data = {
                    'title': song_info.title,
                    'artist': song_info.artist,
                    'album_image_url': song_info.album_image_url,
                    'lyrics': song_info.lyrics
                }
                return JsonResponse(data, status=200)
            else:
                # Song not found
                return JsonResponse({'error': 'Song not found'}, status=404)

        except Exception as e:
            # Handle any other exceptions
            logger.error(f"An error occurred: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)

    else:
        # If not a GET request, return 405 Method Not Allowed
        return JsonResponse({'error': 'Method not allowed'}, status=405)