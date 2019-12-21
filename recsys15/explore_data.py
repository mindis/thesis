
""""
A python file for exploration of recsys-challenge-2015 dataset
"""
import datetime
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/home/nick/Desktop/thesis/datasets/recsys-challenge-2015/processed/train_full_splitted/1.csv',sep ='\t')
# get the shape of the dataset
print(df.head())

#calculate the clicks on each SessionId
session_len = df['SessionId'].value_counts()
session_len = session_len.to_frame().reset_index()
session_len.columns = ['SessionId', 'clicks']
session_len.sort_values(by=['SessionId'],inplace=True)
session_len.set_index('SessionId',inplace=True)
print(session_len)
#plot that
# plt.plot(session_len)
# plt.show()