import pandas as pd
import numpy as np


#Build mappings for indexed users, products, banners
userkey = 'user_id'
itemkey = 'product_id'
timekey = 'timestamp'

df = pd.read_csv('/home/nick/Desktop/thesis/datasets/pharmacy-data/rnn_data.csv')
print(df)

# df.sort_values([userkey, timekey], inplace=True)
#
# userids = df[userkey].unique()
# useridmap = pd.Series(data=np.arange(len(userids)), index=userids)
# useridmap.index.names = ['user_id']
# print(useridmap)
#
# itemids = df[itemkey].unique()
# itemidmap = pd.Series(data=np.arange(len(itemids)), index=itemids)
# itemidmap.index.names = ['product_id']
# print(itemidmap)

