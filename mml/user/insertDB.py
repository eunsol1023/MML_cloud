import csv
from user.models import MMLUserInfo, MMLUserGen  # your_app을 장고 앱 이름으로 바꿔야 합니다.

# CSV 파일 경로
csv_file_path = 'C:\\Users\\gjaischool1\\MML_cloud\\mml\\user\\mml_user_gen.csv'

with open(csv_file_path, newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        user_id = row['USER_ID']
        priority = row['Priority']
        genre = row['Genre']

        # MMLUserInfo 테이블에서 USER_ID에 해당하는 사용자 검색
        try:
            user_info = MMLUserInfo.objects.get(username=user_id)
        except MMLUserInfo.DoesNotExist:
            print(f"User not found: {user_id}")
            continue

        # MMLUserGen 인스턴스 생성 및 저장
        MMLUserGen.objects.create(user_info=user_info, priority=priority, genre=genre)