from django.db import models

class MMLMusicTagHis(models.Model):
    title = models.CharField(max_length=255)
    artist = models.CharField(max_length=255)
    image = models.CharField(max_length=2000)
    user_id = models.CharField(max_length=255)
    input_sentence = models.CharField(max_length=300)

    def __str__(self):
        return f"{self.title} by {self.artist}"