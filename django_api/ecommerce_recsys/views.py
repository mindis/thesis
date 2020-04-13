from django.shortcuts import render
import json
import pandas as pd
import csv
from django.http import HttpResponse
from django.core import exceptions
from django.http import JsonResponse
from ecommerce_recsys.models import ProductInteractions, BannerProduct, TopNBanners, BannerInteractions
from ecommerce_recsys.serializers import ProductInteractionsSerializer, BannerProductSerializer, TopNBannersSerializer, BannerInteractionsSerializer
from rest_framework import viewsets,permissions,views,status
from rest_framework.decorators import parser_classes
from rest_framework.decorators import action
from rest_framework.parsers import JSONParser
from rest_framework.response import Response
from django.http import Http404
from rest_framework.views import APIView

class ProductInteractionsViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows product interactions to be viewed or edited.
    """

    #@parser_classes([JSONParser])
    queryset = ProductInteractions.objects.all()
    serializer_class = ProductInteractionsSerializer

    def list(self, request):
        queryset = ProductInteractions.objects.all()
        serializer = ProductInteractionsSerializer(queryset, many=True)
        #transform serializer data to pandas dataframe
        df = pd.DataFrame.from_dict(serializer.data)
        #df.to_csv('/home/nick/Desktop/thesis/datasets/pharmacy-data/dj_user_product.csv')
        return Response(serializer.data)

    # def retrieve(self, request, *args, **kwargs):
    #     return Response({'something': 'my custom JSON'})

    """POST multiple objects"""
    def get_serializer(self, *args, **kwargs):
        if self.request.method.lower() == 'post':
            data = kwargs.get('data')
            kwargs['many'] = isinstance(data, list)
        return super(ProductInteractionsViewSet, self).get_serializer(*args, **kwargs)


    """DELETE whole model"""

    def delete(self, request):

        queryset = ProductInteractions.objects.all()
        queryset.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

    # @action(methods=['single_delete'], detail=False)
    # def single_delete(self, request, pk, format=None):
    #     snippet = self.get_object(pk)
    #     snippet.delete()
    #     return Response(status=status.HTTP_204_NO_CONTENT)

        # """PUT multiple objects"""
    # @action(methods=['put'],detail = False)
    # def put(self, request, *args, **kwargs):
    #     # partial = kwargs.pop('partial', False)
    #     # instance = self.get_object()
    #     queryset = ProductInteractions.objects.all()
    #     queryset.delete()
    #     serializer = ProductInteractionsSerializer(data=request.data)
    #     if serializer.is_valid():
    #         serializer.save()
    #         return Response(serializer.data,status=status.HTTP_200_OK)
    #     return Response(status=status.HTTP_400_BAD_REQUEST)

class BannerProductViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows banners/products to be viewed or edited.
    """
    queryset = BannerProduct.objects.all()
    serializer_class = BannerProductSerializer

    def get_serializer(self, *args, **kwargs):
        if self.request.method.lower() == 'post':
            data = kwargs.get('data')
            kwargs['many'] = isinstance(data, list)
        return super(BannerProductViewSet, self).get_serializer(*args, **kwargs)

    """DELETE whole model"""
    @action(methods=['delete'], detail=False)
    def delete(self, request):
        queryset = BannerProduct.objects.all()
        queryset.delete()
        return Response(status=status.HTTP_200_OK)

    # def update(self, request, pk=None):
    #     pass

class BannerInteractionsViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows banners/products to be viewed or edited.
    """
    queryset = BannerInteractions.objects.all()
    serializer_class = BannerInteractionsSerializer

    def get_serializer(self, *args, **kwargs):
        if self.request.method.lower() == 'post':
            data = kwargs.get('data')
            kwargs['many'] = isinstance(data, list)
        return super(BannerInteractionsViewSet, self).get_serializer(*args, **kwargs)

    """DELETE whole model"""
    # @action(methods=['delete'], detail=False)
    def delete(self, request):
        queryset = BannerInteractions.objects.all()
        queryset.delete()
        return Response(status=status.HTTP_200_OK)


class TopNBannersViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows product interactions to be viewed or edited.
    """
    queryset = TopNBanners.objects.all()
    serializer_class = TopNBannersSerializer


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

# class BannerProductView(APIView):
#     """
#     API endpoint that allows banners/products to be viewed or edited.
#     """
#     def get(self, request, format=None):
#         obj_bp = BannerProduct.objects.all()
#         serializer = BannerProductSerializer(obj_bp, many=True)
#         return Response(serializer.data)
#
#     def post(self, request, format=None):
#         serializer = BannerProductSerializer(data=request.data)
#         if serializer.is_valid():
#             serializer.save()
#             return Response(serializer.data, status=status.HTTP_201_CREATED)
#         return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)



# class ExampleView(APIView):
#     """
#     A view that can accept POST requests with JSON content.
#     """
#     parser_classes = [JSONParser]
#
#     def post(self, request, format=None):
#         return Response({'received data': request.data})

