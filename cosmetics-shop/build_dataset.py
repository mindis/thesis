import numpy as np
import pandas as pd
import sklearn
from sklearn import cluster
import matplotlib.pyplot as plt
from sklearn import preprocessing

def get_statistics(df):
    # Get event types
    event_types = df['event_type'].unique()
    print(event_types)
    # Get frequences of event types
    event_types_freq = df['event_type'].value_counts()
    print(event_types_freq)
    event_types_freq = pd.DataFrame(event_types_freq)
    event_types_freq.rename(columns={"event_type": "frequencies"}, inplace=True)
    event_types_freq.index.name = 'event_type'
    print(event_types_freq)

    # visualize frequencies
    plt.xlabel('Event Type')
    plt.ylabel('Number of Events in Dataset')
    plt.bar(range(len(event_types_freq)), event_types_freq['frequencies'])
    plt.xticks(range(len(event_types_freq)), event_types, rotation='vertical')
    plt.show()

def build_dataset_for_rnn(df):

    '''duplicate addtocart rows giving a weight of 5 views &
    transaction rows giving a weight of 10 views'''

    df = df.append([df[df.event_type.eq('cart')]] * 2, ignore_index=True)

    df = df.append([df[df.event_type.eq('purchase')]] * 4, ignore_index=True)
    df.sort_values(by=['user_session', 'event_time'], inplace=True)

    df = pd.DataFrame(df)

    return df

def build_dataset_with_ratings(df, num):
    """Rating function --- Rating range 0-5
        rating(i) = view_num(i) * 0.10 + addtocart_num(i) * 0.30 + transaction_num(i) * 0.60"""


    # for i in range(len(user_item_df)):
    for i in range(num):

        if df['event_type'][i] == 'view':
            df['rating'].iloc[i] *= 0.10 * 5
        elif df['event_type'][i] == 'cart':
            df['rating'].iloc[i] *= 0.30 * 5
        elif df['event_type'][i] == 'purchase':
            df['rating'].iloc[i] *= 0.60 * 5

        print(i)

    # df = pd.Dataframe(user_item_df)
    df = pd.DataFrame(df[:(num-1)])
    print(df)
    ratings_df = df.groupby(by=['user_id', 'product_id']).sum()
    ratings_df = pd.DataFrame(data=ratings_df)

    return ratings_df

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

    PATH = '/home/nick/Desktop/thesis/datasets/cosmetics-shop-data/ratings100k.csv'
    data = pd.read_csv(PATH)  # 2019-Nov.csv for November records
    #data.drop(columns='Unnamed: 0',inplace=True)
    print(data)

    data['user_id'] = data.groupby('user_id').ngroup()
    data['product_id'] = data.groupby('product_id').ngroup()
    print(data)
    #data.to_csv(PATH)
    data['rating'] = data['rating'].apply(lambda x: 5 if (x > 5) else x)
    print(data['rating'].value_counts())
    #print(data)
    # data.drop(columns='brand',inplace=True)
    data.to_csv(PATH)
    #data.to_csv('/home/nick/Desktop/thesis/datasets/cosmetics-shop-data/ratings2.csv')
    #lambda x: x * 2 if x < 10 else (x * 3 if x < 20 else x)
    #lambda (T): if (T > 200): return (200 * exp(-T)); else: return (400 * exp(-T))
    #lambda (T):    if (T > 200): return (200 * exp(-T)); else: return (400 * exp(-T))




    # df = pd.read_csv('/home/nick/Desktop/thesis/datasets/cosmetics-shop-data/rnn_test.csv')
    # df.drop(columns='Unnamed: 0', inplace=True)
    # df = df[['event_time','user_id','product_id']]
    # df.sort_values(by=['user_id','event_time'],inplace=True)
    # print(df)
    # df.to_csv('/home/nick/Desktop/thesis/datasets/cosmetics-shop-data/rnn_test2.csv')
    #get_statistics(data)


    # user_item_matrix = data.groupby(['user_id', 'product_id']).event_type.value_counts().to_frame()
    # user_item_matrix = pd.DataFrame(data=user_item_matrix)
    # user_item_matrix.to_csv(path_or_buf='/home/nick/Desktop/thesis/datasets/cosmetics-shop-data/user_item_matrix_full.csv')
    # print(user_item_matrix)
    #
    # user_item_matrix = pd.read_csv('/home/nick/Desktop/thesis/datasets/cosmetics-shop-data/user_item_matrix_full.csv')
    # user_item_matrix.rename(columns={"event_type.1": "rating"}, inplace=True)
    # user_item_matrix['rating'].astype(float)
    # print(user_item_matrix)
    # ratings = build_dataset_with_ratings(user_item_matrix, 100000)
    # ratings.to_csv('/home/nick/Desktop/thesis/datasets/cosmetics-shop-data/ratings100k.csv')
    # print(ratings)








