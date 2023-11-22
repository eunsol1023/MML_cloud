# user/views.py

from datetime import datetime
import logging
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
from django.http import HttpResponse
from django.middleware.csrf import get_token
from django.contrib.auth.forms import AuthenticationForm

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

from django.http import HttpResponse

def home(request):
    if request.user.is_authenticated:
        response = HttpResponse(f"사용자: {request.user.username}이 로그인했습니다")
    else:
        response = HttpResponse("로그인문제발생!")
    return response

@api_view(['POST'])
@permission_classes([AllowAny])
def login_user(request):
    form = AuthenticationForm(request, data=request.data)

    if form.is_valid():
        user = form.get_user()
        login(request, user)  # This automatically sets the session

        # CSRF 토큰 생성 및 응답에 포함
        csrf_token = get_token(request)
        response = JsonResponse({'message': 'Login successful'}, status=200)
        response.set_cookie('csrftoken', csrf_token)
        return response

    else:
        return HttpResponseBadRequest(form.errors.as_json())

logger = logging.getLogger(__name__)

@api_view(['POST'])
def logout_user(request):
    # Log the current user before logout
    logger.info(f"Attempting to log out user: {request.user}")

    # Log out the user, which invalidates the session
    logout(request)

    # Logging the successful logout
    logger.info("Logout successful")

    # Return a response indicating successful logout
    return JsonResponse({'message': 'Logout successful'}, status=200)

