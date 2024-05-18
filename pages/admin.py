from django.contrib import admin

from .models import *
# Register your models here.
@admin.register(ChatHistory)
class ChatHistoryAdmin(admin.ModelAdmin):
    list_display = ("title", "user", "created_at")
    
