from django.contrib import admin
from django.urls import include, path
from polls import views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('', views.upload_file, name='home_view'),
    path("admin/", admin.site.urls),
]


urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)