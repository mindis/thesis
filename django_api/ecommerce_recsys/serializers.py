from rest_framework import serializers
from ecommerce_recsys.models import ProductInteractions, BannerProduct, BannerInteractions, TopNBanners

class ProductInteractionsSerializer(serializers.HyperlinkedModelSerializer):

    class Meta:
        model = ProductInteractions
        fields = ('user_id', 'product_id', 'timestamp','event_type')

class BannerInteractionsSerializer(serializers.HyperlinkedModelSerializer):

    class Meta:
        model = BannerInteractions
        fields = ('user_id','banner_id','banner_pos','timestamp')

class BannerProductSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = BannerProduct
        fields = ('banner_id', 'product_id')

class TopNBannersSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = TopNBanners
        fields = ('banner_id','rank')




