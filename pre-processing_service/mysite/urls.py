from django.contrib import admin
from django.urls import include, path
from polls import views
from django.conf import settings
from django.conf.urls.static import static

from polls.models import Category, Subcategory, Up_Fle
from rest_framework import routers, serializers, viewsets
from rest_framework import permissions

from drf_yasg.views import get_schema_view
from drf_yasg import openapi

# Serializers define the API representation.
class Up_FleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Up_Fle
        fields = '__all__'
          
# ViewSets define the view behavior.
class Up_FleViewSet(viewsets.ModelViewSet):
    queryset = Up_Fle.objects.all()
    serializer_class = Up_FleSerializer

#----------------------------

# Serializers define the API representation.
class CategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = Category
        fields = '__all__'
          
# ViewSets define the view behavior.
class CategoryViewSet(viewsets.ModelViewSet):
    queryset = Category.objects.all()
    serializer_class = CategorySerializer

#----------------------------

# Serializers define the API representation.
class SubcategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = Subcategory
        fields = '__all__'
          
# ViewSets define the view behavior.
class SubcategoryViewSet(viewsets.ModelViewSet):
    queryset = Subcategory.objects.all()
    serializer_class = SubcategorySerializer

# Routers provide an easy way of automatically determining the URL conf.
router = routers.DefaultRouter()
router.register(r'up_fles', Up_FleViewSet)
router.register(r'categorys', CategoryViewSet)
router.register(r'subcategorys', SubcategoryViewSet)

schema_view = get_schema_view(
    openapi.Info(
        title="Your Project Title",
        default_version='v1',
        description="Тестовое описание",
        terms_of_service="https://www.google.com/policies/terms/",
        contact=openapi.Contact(email="contact@snakesandrubies.com"),
        license=openapi.License(name="BSD License"),
    ),
    public=True,
    permission_classes=(permissions.AllowAny,),
)

urlpatterns = [
    path('', views.upload_file, name='home_view'),
    path("admin/", admin.site.urls),
    path('api/', include(router.urls)),
    path('api-auth/', include('rest_framework.urls', namespace='rest_framework')),
    path('swagger/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    path('redoc/', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
]


urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)