from django.db import models
from django.core.validators import RegexValidator
from django.template.defaultfilters import slugify
from django.contrib.auth.models import User

alphanumeric = RegexValidator(r'^[0-9a-zA-Z]*$', 'Only alphanumeric characters are allowed.')

class ProductInteractions(models.Model):

    user_id = models.CharField(max_length=64, blank=False, null=False, validators=[alphanumeric])
    product_id = models.CharField(max_length=64, blank=True, null=False, validators=[alphanumeric])
    cookie_id = models.CharField(max_length=64,blank=True,null=False,validators=[alphanumeric])
    timestamp = models.DateTimeField(auto_now_add=False,null=False)
    event_type = models.CharField(max_length=20,blank=True,null=False)

    class Meta:
        ordering = ['user_id','timestamp']

    # def __str__(self):
    #     return self.product_id

class BannerInteractions(models.Model):

    user_id = models.CharField(max_length=64, blank=False, null=False, validators=[alphanumeric])
    cookie_id = models.CharField(max_length=64,blank=True,null=False,validators=[alphanumeric])
    banner_id = models.IntegerField(blank=True, null=False)
    banner_pos = models.IntegerField(blank=True,null=False)
    timestamp = models.DateTimeField(auto_now_add=False, null=False)
    event_type = models.CharField(max_length=20,blank=True,null=False)


    class Meta:
        ordering = ['user_id','timestamp']


class BannerProduct(models.Model):
    banner_id = models.IntegerField(blank=False,null=False)
    product_id = models.CharField(max_length=64, blank=False, null=False, validators=[alphanumeric])

    class Meta:
        ordering = ['banner_id']

    #product_id = models.ForeignKey(ProductInteractions,on_delete=models.CASCADE)

class BannerLocation(models.Model):
    banner_id = models.IntegerField(blank=False, null=False)
    location_id = models.CharField(max_length=16,blank=True,null=False)

    class Meta:
        ordering = ['location_id']


class TopNBanners(models.Model):
    banner_id = models.IntegerField(blank=False, null=False)
    rank = models.IntegerField(blank=False, null=False)




# class BannerInteractions(models.Model):
#     timestamp = models.DateTimeField(auto_now_add=False)
#     user_id = models.CharField(max_length=100, blank=False, null=False, validators=[alphanumeric])
#     banner_id = models.IntegerField(max_length=10,blank=False,null=False)
#
#     class Meta:
#         ordering = ['timestamp']
