import numpy as np
import pandas as pd
import sklearn
from sklearn import cluster
import matplotlib.pyplot as plt
from sklearn import preprocessing
from clean_data import clean_dataset


def build_dataset_for_rnn(df,userkey,timekey):

    '''duplicate * (weight of views) addtocart and purchase rows '''

    CART_WEIGHT = 2
    PURCHASE_WEIGHT = 4

    df = df.append([df[df.event_type.eq('cart')]] * CART_WEIGHT, ignore_index=True)

    df = df.append([df[df.event_type.eq('purchase')]] * PURCHASE_WEIGHT, ignore_index=True)
    df.sort_values(by=[userkey, timekey], inplace=True)

    df = pd.DataFrame(df)

    return df

def build_dataset_with_ratings(df, userkey, itemkey):

    """Rating function --- Rating range 0-5
        rating(i) = (view_num(i) * 0.10 + addtocart_num(i) * 0.30 + transaction_num(i) * 0.60)*5"""

    event_type_rating = {
        'view': 0.5,
        'cart': 1.5,
        'purchase': 3.0,
    }

    df['rating'] = df['event_type'].apply(lambda x: event_type_rating[x])

    ratings_df = df.groupby([userkey, itemkey]).sum().reset_index()
    ratings_df = pd.DataFrame(ratings_df)

    return ratings_df

def threshold_rating(df, upper_thres = 5):

    df['rating'] = df['rating'].apply(lambda x: upper_thres if (x > upper_thres) else x)

    return df

def change_ui_index(df,userkey,itemkey):

    df[userkey] = df.groupby(userkey).ngroup()
    df[itemkey] = df.groupby(itemkey).ngroup()

    return df


def normalize_ratings(df):

    """Normalize ratings in 0 to 5 scale"""

    x = np.array(df['rating'].values).reshape(-1,1) # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 5))
    x_scaled = min_max_scaler.fit_transform(x)
    x_scaled = np.array(x_scaled)
    df['rating'] = x_scaled.round(2)
    print(df)

    return df


if __name__ == "__main__":

    PATH = '/home/nick/Desktop/thesis/datasets/cosmetics-shop-data/2019-Oct.csv'
    data = pd.read_csv(PATH)

    """Cosmetics e-shop"""
    userkey = 'user_id'  # or 'user_session'
    itemkey = 'product_id'
    timekey = 'event_time'
    brandkey = 'brand'

    data = data[data.event_type != 'remove_from_cart']
    data.drop(columns=['price', 'category_id', 'category_code'], inplace=True)
    data = data.dropna(subset=['brand'])


    print(data.shape)
    data = clean_dataset(data,userkey,itemkey,timekey)
    # print(data)
    print(data.shape)

    #BUILD IMPLICIT RATINGS
    ratings_df = build_dataset_with_ratings(data, userkey, itemkey)
    print(ratings_df)
    # ratings_df.to_csv('/home/nick/Desktop/thesis/datasets/cosmetics-shop-data/implicit-data/implicit_ratings.csv')
    #
    # ratings_df_thr5 = threshold_rating(ratings_df)
    # ratings_df_thr5.to_csv('/home/nick/Desktop/thesis/datasets/cosmetics-shop-data/implicit-data/implicit_ratings_thr5.csv')

    # #binarize_interactions
    # ratings_df['rating'] = 1
    # ratings_df.to_csv('/home/nick/Desktop/thesis/datasets/cosmetics-shop-data/implicit-data/implicit_binarized.csv')