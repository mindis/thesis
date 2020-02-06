import numpy as np
import pandas as pd

from explore_dataset import events_explored

EVENTS_PATH = '/home/nick/Desktop/thesis/datasets/retail-rocket/events.csv'

events,visitors = events_explored(EVENTS_PATH)

events.sort_values(by = 'visitorid', inplace=True)
print(events)
print(events['event'].unique())
#print(events['event'][1])


#events = pd.DataFrame(events.head(1000))
events.reset_index(inplace=True)
events.drop(axis=1, columns='index',inplace=True)
cart_df = pd.DataFrame(events.loc[events['event'] == 'addtocart'])
print(len(cart_df))
print(len(events))
print(events.iloc[1])

# duplicate addtocart rows giving a weight of 5 views
df = events.append([events[events.event.eq('addtocart')]]*4,ignore_index=True)
# duplicate transaction rows giving a weight of 10 views
df2 = df.append([df[df.event.eq('transaction')]]*9,ignore_index=True)
df2.sort_values(by=['visitorid','timestamp'],inplace=True)

df2 = pd.DataFrame(df2)
df2.drop(axis=1,columns='transactionid',inplace=True)

print(df2)



#delete rare items
items_freq = pd.DataFrame(df2.itemid.value_counts())
items_freq.reset_index(inplace=True)
items_freq.rename(columns={"index": "itemid", "itemid": "freq"},inplace=True)
print(items_freq)

items_freq = items_freq[items_freq['freq'] == 1]
list = items_freq['itemid']
print(list)
#keep only those records with 2 or more actions
df2 = df2[~df2['itemid'].isin(list)]
print(df2)

#delete single view registers
visitors_freq = pd.DataFrame(df2.visitorid.value_counts())
visitors_freq.reset_index(inplace=True)
visitors_freq.rename(columns={"index": "visitorid", "visitorid": "freq"},inplace=True)
print(visitors_freq)

visitors_freq = visitors_freq[(visitors_freq['freq'] == 1) | (visitors_freq['freq'] > 30)]
list = visitors_freq['visitorid']
print(list)
#keep only those records with 2 or more actions
df2 = df2[~df2['visitorid'].isin(list)]
print(df2)

cleaned_data = pd.DataFrame(data=df2)
cleaned_data.drop(axis=1,columns='event',inplace=True)
#extract to csv
cleaned_data.to_csv(path_or_buf='/home/nick/Desktop/thesis/datasets/retail-rocket/preprocessed_data.csv')






