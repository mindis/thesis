
""""
A python file for exploration of recsys-challenge-2015 dataset
"""
import datetime
import datetime as dt
import pandas as pd
from datetime import timedelta
from datetime import datetime
import matplotlib.pyplot as plt


DATAPATH = '/media/nick/EA6E30976E305E8F/Users/user/Desktop/thesis_data/recsys-challenge-2015/processed/train_full_splitted/1.csv'
df = pd.read_csv(DATAPATH,sep ='\t')
#unix timestamp to datetime
df['Time']=(pd.to_datetime(df['Time'],unit='s'))
# get the shape of the dataset
print(df.head(10))

df.Time = pd.to_datetime(df.Time)
session_duration_df = df.groupby('SessionId')['Time'].agg(lambda x: max(x) - min(x)).to_frame()
session_duration_df = pd.DataFrame(session_duration_df)
print(session_duration_df)

#calculate the clicks on each SessionId
session_len = df['SessionId'].value_counts()
session_len = session_len.to_frame().reset_index()
session_len.columns = ['SessionId', 'clicks']
session_len.sort_values(by=['SessionId'],inplace=True)
session_len.set_index('SessionId',inplace=True)
print(session_len)

#dataframe with cols ['session_id' , 'clicks']
session_len = pd.DataFrame(data=session_len)
session_len.sort_values(by=['clicks'],ascending=False,inplace=True)
#display the sessions that have more than 20 clicks
print(session_len[session_len['clicks']>20])
#plot that
# plt.plot(session_len)
# plt.show()