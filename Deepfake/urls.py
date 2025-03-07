"""
URL configuration for Deepfake project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from django.conf import settings
from django.conf.urls.static import static

from face import views as userviews

urlpatterns = [
    path('admin/', admin.site.urls),
     path('user-login/', userviews.user_login, name='user_login'),
    path('', userviews.user_register, name='register'),
     path('user-dashboard/', userviews.user_dashboard, name='user_dashboard'),
    path('image-detection/', userviews.image_detection, name='image_detection'),
    path('video-detection/', userviews.video_detection, name='video_detection'),
    path('profile/', userviews.profile, name='profile'),
    path('logout',userviews.user_logout,name="log_out"),


]
