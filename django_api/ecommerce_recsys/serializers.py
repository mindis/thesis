from rest_framework import serializers
from ecommerce_recsys.models import ProductInteractions, BannerProduct, BannerInteractions, BannerLocation, TopNBanners

class ProductInteractionsSerializer(serializers.HyperlinkedModelSerializer):

    class Meta:
        model = ProductInteractions
        fields = ('user_id', 'product_id','cookie_id','timestamp','event_type')

class BannerInteractionsSerializer(serializers.HyperlinkedModelSerializer):

    class Meta:
        model = BannerInteractions
        fields = ('user_id','cookie_id','banner_id','banner_pos','timestamp','event_type')

class BannerProductSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = BannerProduct
        fields = ('banner_id', 'product_id')

class BannerLocationSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = BannerLocation
        fields = ('banner_id', 'location_id')

class TopNBannersSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = TopNBanners
        fields = ('banner_id','rank')




