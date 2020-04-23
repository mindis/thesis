import pandas as pd
import numpy as np

def change_ui_index(df,userkey,itemkey):

    #df[userkey] = df.groupby(userkey).ngroup()
    df[itemkey] = df.groupby(itemkey).ngroup()

    return df

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

    # set keys
    userkey = 'user_id'
    itemkey = 'product_id'
    timekey = 'timestamp'
    date_split = '2017-02-28 23:59:59'

    df = pd.read_csv('/home/nick/Desktop/thesis/datasets/pharmacy-data/product_interactions.csv')

    df = change_ui_index(df,userkey=userkey,itemkey=itemkey)

    df.sort_values(by=timekey, inplace=True)
    print(df[timekey])

    """1) split rnn dataset into training set and testing set based on chronological sequence"""
    train_df, test_df = train_test_split(df,timekey,date_split)
    print(train_df.shape,test_df.shape)

    """2) match trainset with testset items"""
    test_df = match_train_test_items(train_df,test_df,itemkey)
    print(test_df.shape)

    """3) clear single-action sessions from test set"""
    test_df = drop_single_test_sessions(test_df,userkey)
    print(test_df.shape)

    train_df.to_csv('/home/nick/Desktop/thesis/datasets/pharmacy-data/rnn-data/trainset.csv')
    test_df.to_csv('/home/nick/Desktop/thesis/datasets/pharmacy-data/rnn-data/testset.csv')

    print(test_df['user_id'].value_counts())







