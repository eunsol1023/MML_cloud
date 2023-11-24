from django.apps import AppConfig
from django.conf import settings
from joblib import load
import os
import logging

logger = logging.getLogger(__name__)  # 로깅을 위한 설정

class MusicConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "music"
    model = None

    def ready(self):
        # 모델 파일 경로를 settings를 통해 관리
        model_path = os.path.join(settings.BASE_DIR, 'music', 'files', 'word2vec_model.pkl')

        # 파일 경로 존재 여부 및 로딩 과정에서의 오류 처리
        if os.path.exists(model_path):
            try:
                self.model = load(model_path)
                logger.info("Model loaded successfully")  # 성공적인 로딩 로그
            except Exception as e:
                logger.error(f"Error loading the model: {e}")  # 오류 발생 시 로그
        else:
            logger.error(f"Model file not found at {model_path}")  # 파일 미발견 시 로그
