import numpy as np
import pandas as pd
import seaborn as sns # visualizations
import os
import matplotlib.pyplot as plt
from datetime import datetime
import datetime as dt


def clean_dataset(data,userkey,itemkey,timekey):

    """CLEAN DATASET """

    '''1) Drop unnecessary columns of pandas Dataframe / Non Available Values / Duplicates'''
    #data.drop(columns=['price' , 'category_code'],inplace=True)
    #data.dropna(inplace=True)
    #print(data.head(10))
    data.drop_duplicates(inplace=True)

    #sort dataset by user session and event time
    data.sort_values(by = [userkey,timekey],inplace=True)

    '''2) Drop session ids or user ids with a single action'''
    session_freq = data[userkey].value_counts()
    #print(session_freq)
    session_freq = pd.DataFrame({userkey : session_freq.index, 'number of events':session_freq.values})
    session_freq = session_freq[session_freq['number of events'] == 1]
    list = session_freq[userkey]
    data = data[~data[userkey].isin(list)]
    #print(data[userkey].value_counts())


    """ 3) Delete rare product_ids """
    product_freq = data[itemkey].value_counts()
    #print(product_freq)
    product_freq = pd.DataFrame({itemkey:product_freq.index, 'product_frequency':product_freq.values})
    product_freq = product_freq[product_freq['product_frequency'] == 1]
    list2 = product_freq['product_id']
    data = data[~data['product_id'].isin(list2)]
    #print(data[itemkey].value_counts())

    return data
