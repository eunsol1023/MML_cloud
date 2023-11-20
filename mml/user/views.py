# user/views.py

from datetime import datetime
import json
from dateutil.relativedelta import relativedelta
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.contrib.auth import get_user_model
from .serializers import MMLUserInfoSerializer
from django.contrib.auth import authenticate, login
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, HttpResponseBadRequest
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.permissions import AllowAny
from django.utils import timezone
from django.contrib.auth import logout
from django.http import JsonResponse

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

@api_view(['POST'])
@permission_classes([AllowAny])  # 모든 사용자의 접근을 허용합니다.
def login_user(request):
    if request.method == 'POST':
        username = request.data.get('username')
        password = request.data.get('password')
        print(username)
        print(password)
        if not username or not password:
            return HttpResponseBadRequest('사용자 이름과 비밀번호를 모두 제공해야 합니다.')

        user = authenticate(request._request, username=username, password=password)
        if user is not None:
            user.last_login = timezone.now()  # last_login을 현재 시간으로 업데이트합니다.
            user.save(update_fields=['last_login'])
            login(request._request, user)
            return JsonResponse({'message': '로그인 성공'}, status=200)
        else:
            return JsonResponse({'error': '로그인 실패'}, status=401)
    else:
        return JsonResponse({'error': 'POST 요청이 아닙니다'}, status=400)

@api_view(['POST'])
def logout_user(request):
    logout(request)
    return JsonResponse({'message': '로그아웃 성공'}, status=200)