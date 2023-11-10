# users/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('signup/', views.signup, name='signup'),
    # 여기에 더 많은 사용자 관련 URL 패턴을 추가할 수 있습니다.
]
