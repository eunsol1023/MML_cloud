from rest_framework import serializers
from .models import MMLUserInfo

class MMLUserInfoSerializer(serializers.ModelSerializer):
    class Meta:
        model = MMLUserInfo
        fields = ['id', 'username', 'password', 'gender', 'age_range', 'is_active', 'is_staff', 'date_joined', 'last_login']
        extra_kwargs = {
            'password': {'write_only': True},
            'last_login': {'read_only': True},
            'date_joined': {'read_only': True},
            'id': {'read_only': True}
        }

    def create(self, validated_data):
        # create_user 메서드를 사용하여 비밀번호를 해시하고 새 사용자 인스턴스를 생성합니다.
        user = MMLUserInfo.objects.create_user(
            username=validated_data['username'],
            password=validated_data['password'],
            gender=validated_data.get('gender'),
            age_range=validated_data.get('age_range'),
            is_active=validated_data.get('is_active', True),
            is_staff=validated_data.get('is_staff', False)
        )
        return user

    def update(self, instance, validated_data):
        # 비밀번호를 제외한 필드를 업데이트합니다.
        for field, value in validated_data.items():
            if field == 'password':
                instance.set_password(value)
            else:
                setattr(instance, field, value)
        instance.save()
        return instance
