import numpy as np
import pandas as pd
from sklearn import preprocessing


def rate_data(df , num):
    """Rating function --- Rating range 0-5
    rating(i) = view_num(i) * 0.10 + addtocart_num(i) * 0.30 + transaction_num(i) * 0.60"""

    # for i in range(len(user_item_df)):
    for i in range(num):

        if df['event'][i] == 'view':
            df['rating'].iloc[i] *= 0.10 * 5
        elif df['event'][i] == 'addtocart':
            df['rating'].iloc[i] *= 0.30 * 5
        elif df['rating'][i] == 'transaction':
            df['rating'].iloc[i] *= 0.60 * 5

        # user_item_df['rating'].iloc[i] = min(user_item_df['rating'].iloc[i],5)

        print(i)

    # df = pd.Dataframe(user_item_df)
    df = pd.DataFrame(df[:num])
    print(df)
    ratings_df = df.groupby(by=['visitorid', 'itemid']).sum()
    ratings_df = pd.DataFrame(data=ratings_df)

    return ratings_df

def change_index(df):

    df['new_index'] = 0
    print(df)

    df['new_index'].iloc[0] = df['visitorid'].iloc[0]
    counter = df['visitorid'].iloc[0]

    for i in range(1, len(df)):

        if df['visitorid'].iloc[i] != df['visitorid'].iloc[i - 1]:
            counter += 1

        df['new_index'].iloc[i] = counter

    df['visitorid'] = df['new_index']
    df.drop(axis=1, columns='new_index', inplace=True)

    return df


if __name__ == "__main__":

    user_item_df = pd.read_csv(filepath_or_buffer='/home/nick/Desktop/thesis/datasets/retail-rocket/user_item_interactions.csv')

    user_item_df.rename(columns={"event.1": "rating"}, inplace=True)
    user_item_df['rating'].astype(float)

    #Give ratings
    RECORDS = 100
    ratings_df = rate_data(user_item_df,RECORDS)

    print(ratings_df)
    # ratings_df.to_csv(path_or_buf='/home/nick/Desktop/thesis/datasets/retail-rocket/ratings.csv')

    ratings = pd.read_csv('/home/nick/Desktop/thesis/datasets/retail-rocket/ratings.csv')
    ratings = ratings[:RECORDS]
    #change the index
    ratings = change_index(ratings)
    print(ratings)






