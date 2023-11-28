from django.apps import AppConfig
from joblib import load
import os

class MusicConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"

    name = "music"

    model_path = './music/files/word2vec_model.pkl'
    model = None

    def ready(self):
        print('Ready 구문이 실행중입니다')
        if os.path.exists(self.model_path):
            self.model = load(self.model_path)