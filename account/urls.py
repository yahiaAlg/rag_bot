from django.urls import path,include
from django.contrib.auth import views as auth_views
from . import views
urlpatterns = [
    # path("login/", views.user_login, name="login"),
    path("", views.dashboard, name="dashboard"),
    path("login/", views.user_login, name="login"),
    path("register/", views.register, name="register"),
    path("logout/", views.logout, name="logout"),
    path("delete-account/", views.delete_account, name="delete_account"),
    path("profile/", views.dashboard, name="profile"),
    # change password urls
    path('password-change/',
    views.password_change,
    name='password_change'),
    path('password-change/done/',auth_views.PasswordChangeDoneView.as_view(),name='password_change_done'),

]
