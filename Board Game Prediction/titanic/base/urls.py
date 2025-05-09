# base/urls.py
from django.urls import path
from . import views # Your views
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', views.landing_page, name='landing_page'), # Root URL now points to the landing page
    path('predict-tool/', views.prediction_tool_view, name='prediction_tool'), # New URL for the tool
    path('result/', views.predict, name='predict_submit'), # Renamed for clarity
    path('login/', auth_views.LoginView.as_view(template_name='login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='landing_page'), name='logout'), # Redirect to landing_page
    path('register/', views.register, name='register'),
]