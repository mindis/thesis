import numpy as np
import pandas as pd
from sklearn import preprocessing



user_item_df = pd.read_csv(filepath_or_buffer='/home/nick/Desktop/thesis/datasets/retail-rocket/user_item_interactions.csv')

user_item_df.rename(columns={"event.1": "rating"}, inplace=True)
user_item_df['rating'].astype(float)


"""Rating function --- Rating range 0-5
rating(i) = view_num(i) * 0.10 + addtocart_num(i) * 0.30 + transaction_num(i) * 0.60"""
#
#for i in range(len(user_item_df)):
for i in range(10000):

    if user_item_df['event'][i] == 'view':
        user_item_df['rating'].iloc[i] *= 0.10*5
    elif user_item_df['event'][i] == 'addtocart':
        user_item_df['rating'].iloc[i] *= 0.30*5
    elif user_item_df['rating'][i] == 'transaction':
        user_item_df['rating'].iloc[i] *= 0.60*5

    #user_item_df['rating'].iloc[i] = min(user_item_df['rating'].iloc[i],5)

    print(i)

#df = pd.Dataframe(user_item_df)
df = pd.DataFrame(user_item_df[:10000])
print(df)
ratings_df = df.groupby(by = ['visitorid','itemid']).sum()
ratings_df = pd.DataFrame(data=ratings_df)
print(ratings_df)
ratings_df.to_csv(path_or_buf='/home/nick/Desktop/thesis/datasets/retail-rocket/ratings.csv')

# x = np.array(ratings_df[['rating']].values) #returns a numpy array
# print(x)
# min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,5))
# x_scaled = min_max_scaler.fit_transform(x)
# print(x_scaled)
# ratings_df[['rating']] = x_scaled
# print(ratings_df)

#data = user_item_df.groupby(by = 'event').rating.apply(lambda x : x*0.15 if x == 'view' else (x*0.3 if x == 'addtocart' else x*0.55))
# data = user_item_df.groupby(by = 'event').rating.apply(lambda x : x*0.15)
# print(data)



