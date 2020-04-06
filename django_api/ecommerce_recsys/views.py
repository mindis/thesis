from django.shortcuts import render
import json
from django.http import JsonResponse
from ecommerce_recsys.models import ProductInteractions, BannerProduct, TopNBanners
from ecommerce_recsys.serializers import ProductInteractionsSerializer, BannerProductSerializer, TopNBannersSerializer
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

    """POST multiple objects"""
    def get_serializer(self, *args, **kwargs):
        if self.request.method.lower() == 'post':
            data = kwargs.get('data')
            kwargs['many'] = isinstance(data, list)
        return super(ProductInteractionsViewSet, self).get_serializer(*args, **kwargs)

    """DELETE whole model"""
    @action(methods=['delete'], detail=False)
    def delete(self, request):

        queryset = ProductInteractions.objects.all()
        queryset.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


    #@action(methods=['put'],detail=False)
    def update(self, request, *args, **kwargs):
        # partial = kwargs.pop('partial', False)
        # instance = self.get_object()
        serializer = ProductInteractionsSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data,status=status.HTTP_200_OK)
        return Response(status=status.HTTP_400_BAD_REQUEST)

        # serializer = self.get_serializer(instance, data=request.data, partial=partial)
        # serializer.is_valid(raise_exception=True)
        # self.perform_update(serializer)


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

    def update(self, request, *args, **kwargs):
        # partial = kwargs.pop('partial', False)
        # instance = self.get_object()
        serializer = ProductInteractionsSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data,status=status.HTTP_200_OK)
        return Response(status=status.HTTP_400_BAD_REQUEST)

class TopNBannersViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows product interactions to be viewed or edited.
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



# class ExampleView(APIView):
#     """
#     A view that can accept POST requests with JSON content.
#     """
#     parser_classes = [JSONParser]
#
#     def post(self, request, format=None):
#         return Response({'received data': request.data})

