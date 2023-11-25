from rest_framework import serializers
from .models import MMLUserInfo, MMLUserGen, MMLUserLikeArtist

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
        # 사용자 인스턴스 생성
        user = MMLUserInfo.objects.create_user(
            username=validated_data['username'],
            password=validated_data['password'],
            gender=validated_data.get('gender'),
            age_range=validated_data.get('age_range'),
            is_active=validated_data.get('is_active', True),
            is_staff=validated_data.get('is_staff', False)
        )

        # 장르 데이터 처리
        genre_priority_data = validated_data.pop('genre_priority', {})
        for priority, genre in genre_priority_data.items():
            MMLUserGen.objects.create(
                username=user,
                genre=genre,
                priority=priority
            )

        # 아티스트 데이터 처리
        for i in range(1, 6):
            artist = validated_data.pop(f'artist{i}', None)
            if artist:
                MMLUserLikeArtist.objects.create(
                    gen=user.gender,
                    age_group=user.age_range,
                    artist_id=artist,
                    user_id=user.username
                )
                
        return user

    def update(self, instance, validated_data):
        # 필드 업데이트
        for field, value in validated_data.items():
            if field == 'password':
                instance.set_password(value)
            else:
                setattr(instance, field, value)
        instance.save()
        return instance

class MMLUserGenSerializer(serializers.ModelSerializer):
    class Meta:
        model = MMLUserGen
        fields = ['username', 'genre', 'priority']