# music/app.py

from django.apps import AppConfig
from joblib import load
import os
class MusicConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "music"
    model_path = './music/files/word2vec_model.pkl'
    model = None
    def ready(self):
        if os.path.exists(self.model_path):
            self.model = load(self.model_path)