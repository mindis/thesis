import numpy as np
import pandas as pd

from data_exploration import events_explored


def clean_dataset(events):
    events.sort_values(by='visitorid', inplace=True)
    print(events)
    print(events['event'].unique())

    # events = pd.DataFrame(events.head(1000))
    events.reset_index(inplace=True)
    events.drop(axis=1, columns='index', inplace=True)
    cart_df = pd.DataFrame(events.loc[events['event'] == 'addtocart'])
    print(len(cart_df))
    #print(len(events))
    #print(events.iloc[1])


    # '''duplicate addtocart rows giving a weight of 5 views &
    # transaction rows giving a weight of 10 views'''
    # df = events.append([events[events.event.eq('addtocart')]] * 4, ignore_index=True)
    #
    # df2 = df.append([df[df.event.eq('transaction')]] * 9, ignore_index=True)
    # df2.sort_values(by=['visitorid', 'timestamp'], inplace=True)
    #
    # df2 = pd.DataFrame(df2)
    # df2.drop(axis=1, columns='transactionid', inplace=True)
    #
    # print(df2)

    '''delete rare items'''
    items_freq = pd.DataFrame(events.itemid.value_counts())
    items_freq.reset_index(inplace=True)
    items_freq.rename(columns={"index": "itemid", "itemid": "freq"}, inplace=True)
    print(items_freq)

    items_freq = items_freq[items_freq['freq'] == 1]
    list = items_freq['itemid']
    print(list)
    # keep only those records with 2 or more actions
    events = events[~events['itemid'].isin(list)]
    print(events)

    '''keep visitors with >1 & <=30 clicks'''

    visitors_freq = pd.DataFrame(events.visitorid.value_counts())
    visitors_freq.reset_index(inplace=True)
    visitors_freq.rename(columns={"index": "visitorid", "visitorid": "freq"}, inplace=True)
    print(visitors_freq)

    visitors_freq = visitors_freq[(visitors_freq['freq'] == 1) | (visitors_freq['freq'] > 30)]
    list = visitors_freq['visitorid']
    print(list)
    # keep only those records with 2 or more actions
    events = events[~events['visitorid'].isin(list)]
    print(events)

    cleaned_data = pd.DataFrame(data=events)
    cleaned_data.drop(axis=1, columns='event', inplace=True)

    return cleaned_data

if __name__ == "__main__":


    EVENTS_PATH = '/home/nick/Desktop/thesis/datasets/retail-rocket/events.csv'

    events,visitors = events_explored(EVENTS_PATH)

    final_df = clean_dataset(events)
    print(final_df)

    #print(final_df.groupby('visitorid').size())


    #extract to csv
    final_df.to_csv(path_or_buf='/home/nick/Desktop/thesis/datasets/retail-rocket/preprocessed_data.csv')






