from django.shortcuts import render
from rest_framework import APIView
from rest_framework.response import Response
from rest_framework import status

from .serializers import *


# Create your views here.
class BaboView(APIView):
    def post(self, request):
        asdf = model.object.all()
        result = Serializers(instance=asdf)
        
        return Response({'message':'바보진수'}, status=status.HTTP_201_CREATED)