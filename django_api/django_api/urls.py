"""django_api URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
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
from django.urls import path
from django.contrib import admin
#from . import views
from django.urls import include, path
from rest_framework import routers
from ecommerce_recsys import views


router = routers.DefaultRouter()
router.register(r'product-interactions', views.ProductInteractionsViewSet)
router.register(r'relations', views.BannerProductViewSet)
router.register(r'banner-interactions',views.BannerInteractionsViewSet)
router.register(r'banners-recommendation', views.TopNBannersViewSet)


urlpatterns = [
    path('', include(router.urls)),
    path('export/user-product', views.export_user_product),
    path('export/banner-product',views.export_banner_product),
    path('admin/', admin.site.urls)
    #path('api-auth/', include('rest_framework.urls', namespace='rest_framework'))
]