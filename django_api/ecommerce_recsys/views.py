import csv
from django.http import HttpResponse
from rest_framework.views import APIView
from ecommerce_recsys.models import ProductInteractions, BannerProduct, TopNBanners, BannerInteractions, BannerLocation



"""Export json data stored in tables in csv files"""

def export_user_product(request):

    response = HttpResponse(content_type='text/csv')

    writer = csv.writer(response)
    writer.writerow(['user_id', 'product_id', 'timestamp','event_type'])

    for record in ProductInteractions.objects.all().values_list('user_id', 'product_id', 'timestamp','event_type'):
        writer.writerow(record)

    response['Content-Disposition'] = 'attachment; filename="user_product.csv"'

    return response


def export_banner_product(request):

    response = HttpResponse(content_type='text/csv')

    writer = csv.writer(response)
    writer.writerow(['banner_id', 'product_id'])

    for record in BannerProduct.objects.all().values_list('banner_id', 'product_id'):
        writer.writerow(record)

    response['Content-Disposition'] = 'attachment; filename="banner_product.csv"'

    return response