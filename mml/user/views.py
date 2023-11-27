# user/views.py

from datetime import datetime
import logging
from dateutil.relativedelta import relativedelta
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
from django.contrib.auth import get_user_model
from .serializers import MMLUserInfoSerializer
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login as auth_login, logout as auth_logout
from django.http import JsonResponse
from django.utils.decorators import method_decorator
from django.middleware.csrf import get_token
from .serializers import MMLUserInfoSerializer, MMLUserGenSerializer, MMLUserLikeArtist


# Create a logger instance
logger = logging.getLogger(__name__)

User = get_user_model()

@api_view(['POST'])
def signup(request):
    data = request.data
    print(data)
    # 나이 범위 계산
    if data.get('age_range'):
        birthdate = datetime.strptime(data['age_range'], "%Y-%m-%d")
        today = datetime.now()
        age = relativedelta(today, birthdate).years
        if 10 <= age < 20:
            age_range = "teenagers"
        elif 20 <= age < 30:
            age_range = "20s"
        elif 30 <= age < 40:
            age_range = "30s"
        elif 40 <= age < 50:
            age_range = "40s"
        elif 50 <= age < 60:
            age_range = "50s"
        elif 60 <= age < 70:
            age_range = "60s"
        else:
            age_range = "Other age range"
        data['age_range'] = age_range

    # 사용자 데이터 직렬화 및 저장
    serializer = MMLUserInfoSerializer(data=data)
    if serializer.is_valid():
        user = serializer.save()

        # 장르 및 우선순위 데이터 처리
        genre_priority_data = {
            "1": data.get('genre1'),
            "2": data.get('genre2'),
            "3": data.get('genre3'),
            "4": data.get('genre4'),
            "5": data.get('genre5'),
        }
        for priority, genre in genre_priority_data.items():
            if genre:
                mml_user_gen_serializer = MMLUserGenSerializer(data={
                    'username': user,
                    'genre': genre,
                    'priority': priority
                })
                if mml_user_gen_serializer.is_valid():
                    mml_user_gen_serializer.save()

        # 아티스트 데이터 처리
        for i in range(1, 6):
            artist = data.get(f'artist{i}')
            if artist:
                MMLUserLikeArtist.objects.create(
                    gen=user.gender,
                    age_group=user.age_range,
                    artist_id=artist,
                    user_id=user.username
                )

        return Response(serializer.data, status=status.HTTP_201_CREATED)

    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

logger = logging.getLogger(__name__)

@api_view(['POST'])
@permission_classes([AllowAny])
def login_user(request):
    form = AuthenticationForm(request, data=request.data)
    if form.is_valid():
        user = form.get_user()
        auth_login(request, user)
        logger.info(f'Login successful for user: {user.username}')
        return JsonResponse({'message': 'Login successful'}, status=200)
    else:
        logger.warning(f'Login failed: {form.errors.as_json()}')
        return JsonResponse({'errors': form.errors.get_json_data()}, status=401)

@api_view(['POST'])
def logout_user(request):
    username = request.user.username
    auth_logout(request)
    logger.info(f'Logout successful for user: {username}')
    return JsonResponse({'message': 'Logged out'}, status=200)


