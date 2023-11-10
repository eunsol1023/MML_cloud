from django.urls import path
from .views import BaboView

urlpatterns = [
    path('babo/', BaboView.as_view(), name='babo'),
]
