# visualizer/urls.py
from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/run-search/', views.run_search, name='run_search'),

    # --- ADD THESE NEW URLS ---
    path('api/save-graph/', views.save_graph, name='save_graph'),
    path('api/get-graphs/', views.get_graphs, name='get_graphs'),
    # --- END NEW URLS ---

    # Authentication URLs
    path('register/', views.register_view, name='register'),
    path('login/', auth_views.LoginView.as_view(template_name='visualizer/login.html'), name='login'),
    path('logout/', views.logout_view, name='logout'),
]