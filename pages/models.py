from django.db import models
from django.contrib.auth.models import User
from django.urls import reverse
# Create your models here.
class ChatHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=50, blank=True)
    history = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    class Meta:
        verbose_name = "chat_history"
        verbose_name_plural = "chat_histories"

    def __str__(self):
        return self.title

    def get_absolute_url(self):
        return reverse("chathistory_detail", kwargs={"pk": self.pk})
