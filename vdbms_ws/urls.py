"""vdbms_ws URL Configuration

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
from django.contrib import admin
from django.urls import path
from seiden.views import login, my_obtain_jwt_token
from seiden import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('login/', my_obtain_jwt_token),
    path('users/', views.get_users),
    path('users/query', views.get_query_users),
    path('users/add', views.add_user),
    path('users/delete', views.delete_user_by_id),
    path('users/delete/batch', views.delete_batch_users),
    path('users/update/url', views.update_user_profile_url),

    path('videos/', views.get_videos),
    path('videos/play/<str:video_id>', views.get_video),
    path('videos/delete', views.delete_video_by_id),
    path('videos/delete/batch', views.delete_batch_videos),
    path('upload/', views.upload_video),
    path('videos/query', views.get_query_videos),
    path('videos/add', views.add_video),
    path('videos/id', views.get_video_by_id),
    path('videos/ids', views.get_videos_by_ids),
    path('register/', views.register),

    path('model/', views.get_models),
    path('model/load', views.load_data),
    path('model/exe/pre', views.exe_model_pre),
    path('model/exe/rec', views.exe_model_rec),
    path('model/exe/agg', views.exe_model_agg),
    path('model/result/download', views.download_result),
]
# 允许所有的上传文件media文件被访问
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)