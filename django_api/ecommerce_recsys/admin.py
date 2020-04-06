from django.contrib import admin
from .models import ProductInteractions,BannerProduct
from import_export.admin import ImportExportModelAdmin
admin.site.register(ProductInteractions)
admin.site.register(BannerProduct)


# @admin.site.register(ProductInteractions)
# class ViewAdmin(ImportExportModelAdmin):
#     pass