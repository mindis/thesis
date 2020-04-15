from django.shortcuts import render
import json
import pandas as pd
from django.http import HttpResponse
from django.core import exceptions
from django.http import JsonResponse,Http404
from ecommerce_recsys.models import ProductInteractions, BannerProduct, TopNBanners, BannerInteractions, BannerLocation
from ecommerce_recsys.serializers import ProductInteractionsSerializer, BannerProductSerializer, TopNBannersSerializer,\
    BannerInteractionsSerializer, BannerLocationSerializer
from rest_framework import viewsets,permissions,views,status
from rest_framework.decorators import action
from rest_framework.parsers import JSONParser
from rest_framework.response import Response
from django_filters import rest_framework
from rest_framework import filters
from django_filters import FilterSet



class ProductInteractionsFilter(rest_framework.FilterSet):

    class Meta:
        model = ProductInteractions
        fields = {
            'user_id':['icontains'],
            'product_id':['icontains'],
            'timestamp':['iexact','lte','gte'],
            'event_type':['iexact']
        }

class ProductInteractionsViewSet(viewsets.ModelViewSet):

    """
    API endpoint that allows product interactions to be viewed or edited.
    """

    #@parser_classes([JSONParser])
    queryset = ProductInteractions.objects.all()
    serializer_class = ProductInteractionsSerializer
    filterset_class = ProductInteractionsFilter

    filter_backends = [rest_framework.DjangoFilterBackend]
    #filter_fields = ('user_id', 'product_id','event_type')
    #search_fields = ('user_id', 'product_id','event_type')

    #overwrite GET and return records that contain purchase as event_type
    # def get_queryset(self):
    #     return ProductInteractions.objects.filter(event_type__icontains = 'purchase')

    def list(self, request):
        queryset = ProductInteractions.objects.all()
        serializer = ProductInteractionsSerializer(queryset, many=True)
        #transform serializer data to pandas dataframe
        df = pd.DataFrame.from_dict(serializer.data)
        #print(df)
        #df.to_csv('/home/nick/Desktop/thesis/datasets/pharmacy-data/dj_user_product.csv')
        return Response(serializer.data)

    """Receive multiple POSTS"""
    def get_serializer(self, *args, **kwargs):
        if self.request.method.lower() == 'post':
            data = kwargs.get('data')
            kwargs['many'] = isinstance(data, list)
        return super(ProductInteractionsViewSet, self).get_serializer(*args, **kwargs)


    """DELETE all records"""

    def delete(self, request):

        queryset = ProductInteractions.objects.all()
        queryset.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

    # @action(methods=['single_delete'], detail=False)
    # def single_delete(self, request, pk, format=None):
    #     snippet = self.get_object(pk)
    #     snippet.delete()
    #     return Response(status=status.HTTP_204_NO_CONTENT)

class BannerProductFilter(rest_framework.FilterSet):

    class Meta:
        model = BannerProduct
        fields = {
            'banner_id':['iexact'],
            'product_id':['icontains'],
        }

class BannerProductViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows banners/products to be viewed or edited.
    """
    queryset = BannerProduct.objects.all()
    serializer_class = BannerProductSerializer
    filterset_class = BannerProductFilter

    filter_backends = [rest_framework.DjangoFilterBackend]

    """Receive multiple POSTS"""
    def get_serializer(self, *args, **kwargs):
        if self.request.method.lower() == 'post':
            data = kwargs.get('data')
            kwargs['many'] = isinstance(data, list)
        return super(BannerProductViewSet, self).get_serializer(*args, **kwargs)

    """DELETE all records"""
    def delete(self, request):
        queryset = BannerProduct.objects.all()
        queryset.delete()
        return Response(status=status.HTTP_200_OK)

    # def update(self, request, pk=None):
    #     pass
class BannerInteractionsFilter(rest_framework.FilterSet):

    class Meta:
        model = BannerInteractions
        fields = {
            'user_id':['icontains'],
            'banner_id': ['iexact'],
            'banner_pos':['iexact'],
            'timestamp':['iexact','lte','gte'],
        }

class BannerInteractionsViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows banners-interactions to be viewed or edited.
    """
    queryset = BannerInteractions.objects.all()
    serializer_class = BannerInteractionsSerializer
    filterset_class = BannerInteractionsFilter

    """Receive multiple posts"""
    def get_serializer(self, *args, **kwargs):
        if self.request.method.lower() == 'post':
            data = kwargs.get('data')
            kwargs['many'] = isinstance(data, list)
        return super(BannerInteractionsViewSet, self).get_serializer(*args, **kwargs)

    """DELETE all records"""
    def delete(self, request):
        queryset = BannerInteractions.objects.all()
        queryset.delete()
        return Response(status=status.HTTP_200_OK)

class BannerLocationFilter(rest_framework.FilterSet):

    class Meta:
        model = BannerLocation
        fields = {
            'banner_id': ['iexact'],
            'location_id':['iexact'],
        }

class BannerLocationViewSet(viewsets.ModelViewSet):

    queryset = BannerLocation.objects.all()
    serializer_class = BannerLocationSerializer
    filterset_class = BannerLocationFilter

    """Receive multiple posts"""

    def get_serializer(self, *args, **kwargs):
        if self.request.method.lower() == 'post':
            data = kwargs.get('data')
            kwargs['many'] = isinstance(data, list)
        return super(BannerLocationViewSet, self).get_serializer(*args, **kwargs)

    """DELETE all records"""

    def delete(self, request):
        queryset = BannerLocation.objects.all()
        queryset.delete()
        return Response(status=status.HTTP_200_OK)


class TopNBannersViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows topN banners to be viewed or edited.
    """
    queryset = TopNBanners.objects.all()
    serializer_class = TopNBannersSerializer




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




