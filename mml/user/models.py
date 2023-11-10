from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
from django.db import models
from django.utils import timezone

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
