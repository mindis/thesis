import numpy as np
import pandas as pd
import sklearn
from sklearn import cluster
import matplotlib.pyplot as plt
from sklearn import preprocessing
from clean_data import clean_dataset


def build_dataset_for_rnn(df,userkey,timekey):

    '''duplicate addtocart rows giving a weight of 5 views &
    transaction rows giving a weight of 10 views'''

    df = df.append([df[df.event_type.eq('cart')]] * 2, ignore_index=True)

    df = df.append([df[df.event_type.eq('purchase')]] * 4, ignore_index=True)
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
