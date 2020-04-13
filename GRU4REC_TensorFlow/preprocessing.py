import pandas as pd
import numpy as np


def train_test_split(data,timekey,date_split):

    train_data = data[data[timekey] <= date_split]
    test_data = data[data[timekey] > date_split]
    train_data.sort_values(by='user_id', inplace=True)
    test_data.sort_values(by='user_id', inplace=True)

    return train_data,test_data

def drop_single_test_sessions(data,userkey):

    session_freq = data[userkey].value_counts()
    session_freq = pd.DataFrame({userkey: session_freq.index, 'number of events': session_freq.values})
    session_freq = session_freq[session_freq['number of events'] == 1]
    list = session_freq[userkey]
    data = data[~data[userkey].isin(list)]

    return data

def match_train_test_items(train_data,test_data,itemkey):

    print(test_data.shape)
    unique_products = train_data[itemkey].unique()
    test_data = test_data[test_data[itemkey].isin(unique_products)]
    print(test_data.shape)

    return test_data


if __name__ == '__main__':

    df = pd.read_csv('/home/nick/Desktop/thesis/datasets/pharmacy-data/rnn_data_indexed.csv')
    df.sort_values(by='timestamp', inplace=True)
    df.drop(columns='Unnamed: 0', inplace=True)
    print(df)

    #set keys
    userkey = 'user_id'
    itemkey = 'product_id'
    timekey = 'timestamp'
    date_split = '2017-02-28 23:59:59'

    """1) split rnn dataset into training set and testing set based on chronological sequence"""
    train_df, test_df = train_test_split(df,timekey,date_split)

    """2) clear single-action sessions from test set"""
    test_df = drop_single_test_sessions(test_df,userkey)

    """3) clear single-action sessions from test set"""
    test_df = match_train_test_items(train_df,test_df,itemkey)

    





