import csv
from django.contrib.auth.hashers import make_password
from django.db import transaction
from user.serializers import MMLUserInfoSerializer

# CSV 파일에서 데이터 읽기
with open('./mml_user_info_modified_corrected.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)

    user_info_instances = []  # MMLUserInfo 인스턴스를 저장할 리스트

    for row in reader:
        user_info = MMLUserInfo(
            username=row['USER_NAME'],
            gender=row['GEN'],
            age_range=row['AGE_GROUP'],
            password=make_password(row['password'])  # 비밀번호를 해시하여 저장
        )
        user_info_instances.append(user_info)

    # bulk_create를 사용하여 데이터베이스에 인스턴스들을 저장
    with transaction.atomic():  # 트랜잭션 관리
        MMLUserInfo.objects.bulk_create(user_info_instances)
