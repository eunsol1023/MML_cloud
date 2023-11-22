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


# Create a logger instance
logger = logging.getLogger(__name__)

User = get_user_model()

@api_view(['POST'])
def signup(request):
    """
    Create a new user instance.
    """
    data = request.data

    # age_range 필드가 None이 아닌 경우에 연령대로 변환
    if data.get('age_range'):
        
        birthdate = datetime.strptime((data['age_range']), "%Y-%m-%d")
        print(type(birthdate))
        today = datetime.now()
        age = relativedelta(today, birthdate).years
        if 10 <= age < 20:
            age_range = "10대"
        elif 20 <= age < 30:
            age_range = "20대"
        elif 30 <= age < 40:
            age_range = "30대"
        elif 40 <= age < 50:
            age_range = "40대"
        elif 50 <= age < 60:
            age_range = "50대"
        elif 60 <= age <70:
            age_range = "60대"
        else:
            age_range = "기타연령대"

    # 데이터를 저장할 때 age_range 필드에 연령대 값 설정
    data['age_range'] = age_range

    serializer = MMLUserInfoSerializer(data=data)
    if serializer.is_valid():
        serializer.save()
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
