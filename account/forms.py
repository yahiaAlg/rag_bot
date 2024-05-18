from django import forms
from django.contrib.auth.models import User
class LoginForm(forms.Form):
    username = forms.CharField()
    password = forms.CharField(widget=forms.PasswordInput)

class RegisterForm(forms.ModelForm):
    
    class Meta:
        model = User
        fields = "__all__"
        exclude = ("groups", "is_staff", "is_superuser", "is_active", "user_permissions", "date_joined", "last_login")
