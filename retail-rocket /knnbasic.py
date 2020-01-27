#KNN Surprise from kaggle

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split as  train_test_split_sklearn
import surprise
from surprise.model_selection.split import train_test_split
from surprise.prediction_algorithms.knns import KNNBasic
from surprise import accuracy

import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
FOLDER = '/home/nick/Desktop/thesis/datasets/retail-rocket/'

events = pd.read_csv(FOLDER + 'events.csv')
category_tree = pd.read_csv(FOLDER + 'category_tree.csv')

item_properties_part1 = pd.read_csv(FOLDER + 'item_properties_part1.csv')
item_properties_part2 = pd.read_csv(FOLDER + 'item_properties_part2.csv')
item_properties_part = pd.concat([item_properties_part1, item_properties_part2])

print(events.head())
print(category_tree.head())
print(item_properties_part.head())

data = events[['visitorid','event','itemid']]
info_event_events = events.groupby(by=['event']).size()
print(info_event_events)
print(data.head())



transfrom_rating = []
# view = 1, addtocart = 2, transaction = 3
def transfrom_data(data_raw):
    data = data_raw.copy()
    for event in data.event:
        if(event == 'view'):
            transfrom_rating.append(1)
        if(event == 'addtocart'):
            transfrom_rating.append(2)
        if(event == 'transaction'):
            transfrom_rating.append(3)
    data['rating']= transfrom_rating
    return data[['visitorid','itemid','rating']]
data_surprise = transfrom_data(data)
print(data_surprise.head())

data_view  = data_surprise[data_surprise['rating']==1].reset_index(drop= True)
data_transaction  = data_surprise[data_surprise['rating']==2].reset_index(drop= True)
data_addtocard  = data_surprise[data_surprise['rating']==3].reset_index(drop= True)

data_view_train, data_view_test = train_test_split_sklearn(data_view, test_size= 0.008)
data_transaction_train, data_transaction_test = train_test_split_sklearn(data_transaction, test_size= 0.33)

data_tuning = pd.concat([data_addtocard, data_view_test, data_transaction_test]).sort_values(by = 'rating').reset_index(drop=True)

print("The number item view ", data_tuning[data_tuning['rating']==1].shape[0])
print("The number item tranaction ", data_tuning[data_tuning['rating']==2].shape[0])
print("The number item addtacard ", data_tuning[data_tuning['rating']==3].shape[0])
print(data_tuning.head())

reader = surprise.Reader(rating_scale=(1, 3))
data = surprise.Dataset.load_from_df(data_tuning, reader)
type(data)
trainset, testset = train_test_split(data, test_size=0.25)
type(trainset)
sim_options = {'name': 'cosine',
               'user_based': False
               }
algo_knn_basic = KNNBasic(sim_options=sim_options)
predictions = algo_knn_basic.fit(trainset).test(testset)
result = pd.DataFrame(predictions, columns=['visitor_id', 'item_id', 'base_event', 'predict_event', 'details'])
result.drop(columns = {'details'}, inplace = True)
result['error'] = abs(result['base_event'] - result['predict_event'])
print(result.head())


result['predict_event'].hist(bins= 100, figsize= (20,10))
result[result['base_event']== 1]['predict_event'].hist(bins= 100, figsize= (20,10))
result[result['base_event']== 2]['predict_event'].hist(bins= 100, figsize= (20,10))



mae_model = accuracy.mae(predictions)
rmse_model = accuracy.rmse(predictions)
print(mae_model,rmse_model)

