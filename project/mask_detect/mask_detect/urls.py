"""mask_detect URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
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
from django.conf.urls import url
from django.contrib import admin
from django.urls import path,include
from . import views
from Userlogin.views import *




urlpatterns = [
    url(r'^$', login),
    # 默认网址下配置登录功能
    path('save/',save),
    path('query/',query),
    path('admin/', admin.site.urls),
    path('login/',login,name='login'),
    path('register/',register,name='register'),
    path('register/save',save),
    path('login/query',query),
    path('index/',index),
    path('index/videoOn',videoOn),
    path('index/logout',logout),
    path('forget/',forget,name='forget'),
    path('forget/send',send),
    path('forget/find',find),
]

