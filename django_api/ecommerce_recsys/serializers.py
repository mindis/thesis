from rest_framework import serializers
from ecommerce_recsys.models import ProductInteractions, BannerProduct, TopNBanners

class ProductInteractionsSerializer(serializers.HyperlinkedModelSerializer):

    # def create(self, validated_data):

    class Meta:
        model = ProductInteractions
        fields = ('user_id', 'product_id', 'timestamp')


class BannerProductSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = BannerProduct
        fields = ('banner_id', 'product_id')

class TopNBannersSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = TopNBanners
        fields = ('banner_id','rank')




