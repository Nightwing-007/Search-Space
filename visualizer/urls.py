# visualizer/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/run-search/', views.run_search, name='run_search'), # Add this line
]