from pprint import pprint
from django.shortcuts import render,redirect
from django.http import HttpResponse
from django.contrib import messages

# Create your views here.

from django.http import HttpResponse
from django.shortcuts import render
from django.contrib.auth import authenticate, login, update_session_auth_hash
from django.contrib import auth
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST

from django.contrib.auth.forms import UserCreationForm, PasswordChangeForm
from .forms import *
from pages.models import *
from pages.langchain_pipeline import config_bot

def user_login(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            cd = form.cleaned_data
            user = authenticate(request,
            username=cd['username'],
            password=cd['password'])
            if user is not None:
                if user.is_active:
                    login(request, user)
                    request.session["prepared"], request.session["chunks"] = (
                        config_bot()
                    )
                    return redirect("index")
                else:
                    return HttpResponse('Disabled account')
            else:
                messages.error(request,"Invalid login")
                return redirect("login")
    else:
        form = LoginForm()
    return render(request, 'account/login.html', {'form': form})


@login_required
def password_change(request):
    if request.method == "POST":
        form = PasswordChangeForm(user=request.user, data=request.POST)
        if form.is_valid():
            form.save()
            update_session_auth_hash(request, request.user)
            messages.success(request, "Your password has been updated successfully!")
            return redirect("dashboard")
        else:
            messages.error(request, "Please correct the error below.")
    else:
        form = PasswordChangeForm(request.user)
    return render(request, "account/password_change.html", {"form": form})


@login_required
def dashboard(request):
    history = ChatHistory.objects.filter(user=request.user)
    
    return render(
        request,
        'account/dashboard.html',
        {'section': 'dashboard', "history":history.first() if history.exists() else "no chat history" }
        )

def logout(request):
    auth.logout(request)
    messages.info(request, "logged out")
    if request.session.get("prepared", None):
        del request.session["chunks"]
        del request.session["prepared"]
    return redirect("login")


@login_required
@require_POST
def delete_account(request):
    user = request.user
    user.delete()
    messages.success(request, "Your account has been deleted successfully!")
    return redirect("login")


def register(request):
    form = UserCreationForm()
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            raw_password = form.cleaned_data.get("password1")
            user = authenticate(username=user.username, password=raw_password)
            auth.login(request, user)
            messages.success(
                request, f"successfully registered as {request.user.username}"
            )
            return redirect("index")
        else:
            messages.error(request, "registration failed")

    return render(request, "account/register.html", {"form": form})
