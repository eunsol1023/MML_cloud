from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
from django.db import models
from django.utils import timezone
from django.conf import settings

class MMLUserInfoManager(BaseUserManager):
    def create_user(self, username, password=None, **extra_fields):
        if not username:
            raise ValueError('The given username must be set')
        user = self.model(username=username, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, username, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        
        return self.create_user(username, password, **extra_fields)

class MMLUserInfo(AbstractBaseUser, PermissionsMixin):
    username = models.CharField(max_length=150, unique=True)
    gender = models.CharField(max_length=1, choices=(('M', 'Male'), ('F', 'Female')))
    age_range = models.CharField(max_length=10)
    date_joined = models.DateTimeField(default=timezone.now)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)

    objects = MMLUserInfoManager()

    USERNAME_FIELD = 'username'
    REQUIRED_FIELDS = []

    class Meta:
        db_table = 'mml_user_info'

    def __str__(self):
        return self.username
    
class MMLUserGen(models.Model):
    # `null=True`를 추가하여 새 필드가 NULL 값을 가질 수 있도록 설정합니다.
    # 기존의 'username' 필드가 문자열을 참조하므로, ForeignKey의 `to_field`를 `username`으로 설정합니다.
    username = models.ForeignKey(
        settings.AUTH_USER_MODEL,  # 이는 기본 User 모델을 참조하는 것으로 가정합니다.
        to_field='username', 
        db_column='username', 
        on_delete=models.CASCADE,
        null=True  # 새로운 필드가 NULL 값을 가질 수 있도록 설정
    )
    priority = models.IntegerField()
    genre = models.CharField(max_length=100)

    class Meta:
        db_table = 'mml_user_gen'

    def __str__(self):
        # `username` 필드가 nullable이므로, `username` 필드가 `None`일 때를 처리하는 로직이 필요합니다.
        return f"{self.username.username if self.username else 'None'} - {self.genre} - {self.priority}"

class MMLArtistGen(models.Model):
    artist = models.CharField(max_length=100, primary_key=True)
    priority = models.IntegerField()  # 우선순위
    genre = models.CharField(max_length=100)  # 장르

    class Meta:
        db_table = 'mml_artist_gen'

    def __str__(self):
        return f"{self.artist} - {self.genre}"

class MMLUserLikeArtist(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        to_field='username', 
        on_delete=models.CASCADE,
        db_column='user_id'
    )
    artist = models.ForeignKey(
        MMLArtistGen, 
        to_field='artist',  # 'artist' 필드를 명시적으로 참조
        on_delete=models.CASCADE,
        db_column='artist_id'
    )
    gen = models.CharField(max_length=10)
    age_group = models.CharField(max_length=50)

    class Meta:
        db_table = 'mml_user_like_artist'

    def __str__(self):
        return f"{self.user.username} - {self.artist.artist}"

class MMLUserLikeMusic(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,  # MMLUserInfo 모델을 참조한다고 가정합니다.
        to_field='username',
        on_delete=models.CASCADE,
        db_column='user_id'
    )
    title = models.CharField(max_length=255)
    artist = models.CharField(max_length=100)

    class Meta:
        db_table = 'mml_user_like_music'

    def __str__(self):
        return f"{self.user.username} - {self.title} - {self.artist}"
    
class MMLMusicInfo(models.Model):
    title = models.CharField(max_length=255)
    artist = models.CharField(max_length=255)
    genre = models.CharField(max_length=100)
    lyrics = models.TextField()
    album_image_url = models.URLField()

    class Meta:
        db_table = 'mml_music_info'
        
    def __str__(self):
        return f"{self.title} - {self.artist}"    

class MMLMusicTag(models.Model):
    title = models.CharField(max_length=255)  # ForeignKey 대신 CharField 사용
    artist = models.CharField(max_length=255)
    tag = models.CharField(max_length=100)

    class Meta:
        db_table = 'mml_music_tag'

    def __str__(self):
        return f"{self.title} - {self.tag}"
    
class MMLUserHis(models.Model):
    user = models.CharField(max_length=150)  # ForeignKey 대신 CharField 사용
    title = models.CharField(max_length=255)  # ForeignKey 대신 CharField 사용
    artist = models.CharField(max_length=100)
    genre = models.CharField(max_length=100)
    playtime = models.IntegerField()
    created_at = models.DateTimeField()

    class Meta:
        db_table = 'mml_user_his'

    def __str__(self):
        return f"{self.user} - {self.title} - {self.artist}"