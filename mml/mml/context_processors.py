# context_processors.py

from django.conf import settings

def global_user_context(request):
    return {
        'global_username': request.user.username if request.user.is_authenticated else None
    }