import numpy as np
import pandas as pd
import sklearn
from sklearn import cluster
from preprocess_data import build_dataset

PATH = '/home/nick/Desktop/thesis/datasets/cosmetics-shop-data/'
data = pd.read_csv(PATH + '2019-Oct.csv')  # 2019-Nov.csv for November records
df = build_dataset(data)

user_products = df.groupby('user_id').product_id.value_counts()
print(user_products)
print(type(user_products))



