import os
from django.core.wsgi import get_wsgi_application
from django.utils import timezone
from user.models import MMLUserInfo  # 사용자 모델을 포함한 모델 경로로 대체하세요

# Django 설정 모듈을 로드
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mml.settings")

# Django 앱 초기화
application = get_wsgi_application()

# 사용자 객체 가져오기 (예: username을 사용하여)
try:
    user = MMLUserInfo.objects.get(username='QrDM6lLc')

    # 새 비밀번호 설정
    new_password = '1234'
    user.set_password(new_password)

    # date_joined 필드에 현재 시간 설정
    user.date_joined = timezone.now()

    # 사용자 객체 저장
    user.save()
    print("비밀번호 변경 및 date_joined 필드 업데이트 완료")
except MMLUserInfo.DoesNotExist:
    print("해당 username을 가진 사용자가 존재하지 않습니다.")