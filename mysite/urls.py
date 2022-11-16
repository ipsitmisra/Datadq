"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
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
from django.urls import path,include
from dqapp import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.final, name='final'),
    path('home', views.home, name='home'),
    path('analyze', views.analyze, name='analyze'),
    path('zip1', views.zip1, name='zip1'),
    path('zipanomoly1', views.zipanomoly1, name='zipanomoly1'),
    path('ph1', views.ph1, name='ph1'),
    path('phanomoly1', views.phanomoly1, name='phanomoly1'),
    path('mail1', views.mail1, name='mail1'),
    path('mailanomoly1', views.mailanomoly1, name='mailanomoly1'),
    path('address1', views.address1, name='address1'),
    path('addressanomoly1', views.addressanomoly1, name='addressanomoly1'),
    path('multifile', views.multifile, name='multifile'),
    path('common', views.common, name='common'),
    path('unique', views.unique, name='unique'),
    path('uniquein1', views.uniquein1, name='uniquein1'),
    path('uniquein2', views.uniquein2, name='uniquein2'),
    path('coloutlier', views.coloutlier, name='coloutlier'),
    path('dataoutlier', views.dataoutlier, name='dataoutlier'),
]
