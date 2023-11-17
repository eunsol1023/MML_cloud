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
    id = models.AutoField(primary_key=True)
    artist = models.CharField(max_length=100)
    priority = models.IntegerField()
    genre = models.CharField(max_length=100)
    artist_processed = models.CharField(max_length=100)

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
        on_delete=models.CASCADE,
        db_column='artist_id'
    )
    gen = models.CharField(max_length=10)
    age_group = models.CharField(max_length=50)

    class Meta:
        db_table = 'mml_user_like_artist'

    def __str__(self):
        return f"{self.user.username} - {self.artist.artist}"
    
    


