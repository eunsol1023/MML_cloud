from django.contrib.sessions.models import Session
from django.contrib import admin
from .models import MMLUserInfo

@admin.register(Session)
class SessionAdmin(admin.ModelAdmin):
    list_display = ['session_key', 'expire_date', 'get_decoded']
    readonly_fields = ['session_key', 'expire_date', 'get_decoded']

    def get_decoded(self, obj):
        return obj.get_decoded()


admin.site.register(MMLUserInfo)