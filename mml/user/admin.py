from django.contrib.sessions.models import Session
from django.contrib import admin

@admin.register(Session)
class SessionAdmin(admin.ModelAdmin):
    list_display = ['session_key', 'expire_date', 'get_decoded']
    readonly_fields = ['session_key', 'expire_date', 'get_decoded']

    def get_decoded(self, obj):
        return obj.get_decoded()

from django.contrib import admin
from .models import user

# Register your models here.
admin.site.register(user)