from django.urls import path

from . import views

urlpatterns = [
    path('chat/', views.chat, name='chat'),
    path('plan/', views.plan, name='plan'),
]
