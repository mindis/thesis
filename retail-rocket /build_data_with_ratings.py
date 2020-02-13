import numpy as np
import pandas as pd

from data_exploration import events_explored



def clean_data(df):

    '''delete rare items'''
    items_freq = pd.DataFrame(df.itemid.value_counts())
    items_freq.reset_index(inplace=True)
    items_freq.rename(columns={"index": "itemid", "itemid": "freq"}, inplace=True)
    #print(items_freq)

    items_freq = items_freq[items_freq['freq'] == 1]
    list = items_freq['itemid']
    print(list)
    # keep only those records with 2 or more actions
    df = df[~df['itemid'].isin(list)]
    print(df)

    '''keep visitors with >1 '''

    visitors_freq = pd.DataFrame(df.visitorid.value_counts())
    visitors_freq.reset_index(inplace=True)
    visitors_freq.rename(columns={"index": "visitorid", "visitorid": "freq"}, inplace=True)
    print(visitors_freq)

    visitors_freq = visitors_freq[(visitors_freq['freq'] == 1)]
    list = visitors_freq['visitorid']
    print(list)
    # keep only those records with 2 or more actions
    df = df[~df['visitorid'].isin(list)]
    print(df)

    cleaned_data = pd.DataFrame(data=df)

    return cleaned_data




EVENTS_PATH = '/home/nick/Desktop/thesis/datasets/retail-rocket/events.csv'

events,visitors = events_explored(EVENTS_PATH)
events.reset_index(inplace=True)
events.drop(axis=1, columns=['index','transactionid'], inplace=True)
print(events)

new_df = clean_data(events)
new_df.sort_values(by=['visitorid','timestamp'],inplace=True)
print(new_df)

user_item_interaction = new_df.groupby(['visitorid','itemid']).event.value_counts().to_frame()
user_item_interaction = pd.DataFrame(data=user_item_interaction)
user_item_interaction.to_csv(path_or_buf='/home/nick/Desktop/thesis/datasets/retail-rocket/user_item_interactions.csv')
print(user_item_interaction)
print(type(user_item_interaction))
print(user_item_interaction.iloc[1])

#index = pd.MultiIndex(user_item_interaction.index,names=('visitorid', 'event', 'itemid'))
#print(index)

# rating = []
# for i in range(len(new_df['visitorid'].nunique()):
#     for j in range(len(new_df['itemid'].nunique()):
#
#         rating[i,j] =


#print(user_item_interaction.index)
# user_item_df = user_item_interaction.to_frame()
# print(user_item_df)