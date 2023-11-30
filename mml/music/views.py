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

from .models import MMLMusicTagHis


# Create or get a logger
logger = logging.getLogger(__name__)

def song_info(request):
    if request.method == 'GET':
        # Django automatically decodes these parameters
        title = request.GET.get('title', None)
        artist = request.GET.get('artist', None)

        logger.info(f"Received title: {title}, artist: {artist}")

        if not title or not artist:
            return JsonResponse({'error': 'Missing required parameters'}, status=400)

        try:
            song_info = MMLMusicInfo.objects.filter(title=title, artist=artist).first()

            if song_info:
                data = {
                    'title': song_info.title,
                    'artist': song_info.artist,
                    'album_image_url': song_info.album_image_url,
                    'lyrics': song_info.lyrics
                }
                return JsonResponse(data, status=200)
            else:
                return JsonResponse({'error': 'Song not found'}, status=404)

        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)

    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    
def tag_song2vec_his(request):
    try:
        # 가장 최근에 사용된 서로 다른 'input_sentence' 두 개를 가져옵니다.
        recent_sentences = MMLMusicTagHis.objects.order_by('-id').values('input_sentence').distinct()[:2]

        # 결과를 저장할 리스트
        results = []

        # 각 'input_sentence'에 대해 관련된 노래 정보를 가져옵니다.
        for sentence in recent_sentences:
            songs = MMLMusicTagHis.objects.filter(
                input_sentence=sentence['input_sentence']
            ).order_by('-id')[:2]

            for song in songs:
                results.append({
                    'input_sentence': song.input_sentence,
                    'title': song.title,
                    'artist': song.artist,
                    'image': song.image
                })

        # 결과를 JSON 형식으로 반환합니다. 상태 코드를 200으로 지정합니다.
        return JsonResponse({'results': results}, status=200)

    except Exception as e:
        # 서버 내부 오류 처리
        return JsonResponse({'error': 'Internal Server Error', 'message': str(e)}, status=500)