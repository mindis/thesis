import numpy as np
import pandas as pd
import seaborn as sns # visualizations
import os
import matplotlib.pyplot as plt

PATH = '/home/nick/Desktop/thesis/datasets/cosmetics-shop-data/'

data = pd.read_csv(PATH + '2019-Oct.csv') #2019-Nov.csv for November records
print(data)
#Get event types
event_types = data['event_type'].unique()
print(event_types)
#Get frequences of event types
event_types_freq = data['event_type'].value_counts()
print(event_types_freq)
event_types_freq = pd.DataFrame(event_types_freq)
event_types_freq.rename(columns = {"event_type": "frequencies"}, inplace= True)
event_types_freq.index.name = 'event_type'
print(event_types_freq)

#visualize frequencies
plt.bar(range(len(event_types_freq)), event_types_freq['frequencies'])
plt.xticks(range(len(event_types_freq)), event_types, rotation='vertical')
#plt.show()

# Get only purchases
only_purchases = data.loc[data.event_type == 'purchase']
print(only_purchases)

#Shows the most popular brands (by total sales)
#With brands only
purchases_with_brands = only_purchases.loc[only_purchases.brand.notnull()]
top_sellers = purchases_with_brands.groupby('brand').brand.agg([len]).sort_values(by='len', ascending=False)
print('Top Seller Brands{}\n'.format(top_sellers.head(20)))

#get number of unique items
unique_product_num = data['product_id'].nunique()
print('\nThe number of unique products viewed or purchased is:{}'.format(unique_product_num))

#get number of unique users
unique_users_num = data['user_id'].nunique()
print('\nThe number of unique users in this dataset is:{}'.format(unique_users_num))

#get number of sessions made
sessions_num = data['user_session'].nunique()
print('\nThe number of sessions in this dataset is:{}'.format(sessions_num))

#example: get all events of the same user
#print(data.loc[data.user_id == 10280338].sort_values(by='event_time'))

cat_num = data['category_id'].nunique()
print('\nThe number of unique categories in this dataset is:{}'.format(cat_num))

#clean dataset
data.drop(columns=['price' , 'category_code'],inplace=True)
data.dropna(inplace=True)
print(data.head(10))

#number of events per user
print(data['user_session'].value_counts())

#sort dataset by user session and event time
data.sort_values(by = ['event_time','user_session'],inplace=True)
print(data)

#calculate session length / drop one event sessions
df = data['user_session'].value_counts()
#pd.DataFrame(df,columns=['user_session','session_number'])

df = pd.DataFrame({'user_session':df.index, 'number of events':df.values})
print(df)
print(df[df['number of events']>1])


# for i in range(len(data)):
#     if data['user_session'][i]


