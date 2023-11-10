# user/views.py

from datetime import datetime
from dateutil.relativedelta import relativedelta
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.contrib.auth import get_user_model
from .serializers import MMLUserInfoSerializer

User = get_user_model()

@api_view(['POST'])
def signup(request):
    """
    Create a new user instance.
    """
    data = request.data
    age_range = None

    # age_range 필드가 None이 아닌 경우에 연령대로 변환
    if data.get('age_range'):
        birthdate = datetime.strptime(data['age_range'], "%Y-%m-%d")
        today = datetime.now()
        age = relativedelta(today, birthdate).years
        if 10 <= age < 20:
            age_range = "10대"
        elif 20 <= age < 30:
            age_range = "20대"
        # 여기서 원하는 연령대 조건을 추가로 처리할 수 있습니다.

    # 데이터를 저장할 때 age_range 필드에 연령대 값 설정
    data['age_range'] = age_range

    serializer = MMLUserInfoSerializer(data=data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
