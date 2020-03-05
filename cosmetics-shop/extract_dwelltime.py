import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt


def compute_dwell_time(df):
    times_t = np.roll(df['event_time'], -1)  # Take time row
    times_dt = df['event_time']  # Copy, then displace by one

    diffs = np.subtract(times_t, times_dt)  # Take the pairwise difference

    length = len(df['product_id'])

    # cummulative offset start for each session
    offset_sessions = np.zeros(df['user_session'].nunique() + 1, dtype=np.int32)
    offset_sessions[1:] = df.groupby('user_session').size().cumsum()

    offset_sessions = offset_sessions - 1
    offset_sessions = np.roll(offset_sessions, -1)

    # session transition implies zero-dwell-time
    # note: paper statistics do not consider null entries,
    # though they are still checked when augmenting
    np.put(diffs, offset_sessions, np.zeros((offset_sessions.shape)), mode='raise')

    return diffs


def join_dwell_reps(df, dt, threshold=2000):
    # Calculate d_ti/threshold + 1
    # then add column to dataFrame

    dt //= threshold
    dt += 1
    df['DwellReps'] = pd.Series(dt.astype(np.int64), index=dt.index)

    return df

if __name__ == "__main__":

    PATH = '/home/nick/Desktop/thesis/datasets/cosmetics-shop-data/brand_dataset.csv'


    df = pd.read_csv(PATH)
    df.drop(columns=['Unnamed: 0'],inplace=True)

    df.sort_values(by=['user_session','event_time'],inplace=True)
    df.event_time = pd.to_datetime(df.event_time)
    df['event_time']=df['event_time'].dt.tz_localize(None)
    #df.event_time = datetime.datetime.strptime()
    print(df)

    dw_t = compute_dwell_time(df)
    dw_t = pd.DataFrame(dw_t)
    print(dw_t)
    #
    # # data = join_dwell_reps(df,dw_t)
    # # print(data)
    """Clean outliers"""
    #dw_t = dw_t[dw_t['timestamp'] <= '1 days 00:00:00']
    # #print(dw_t)
    #
    # """Calculate the average dwelltime of the records, ignoring outliers"""
    # non_zero_dwt = dw_t[(dw_t['timestamp'] != '0 days 00:00:00') & (dw_t['timestamp'] < '0 days 01:00:00')]
    # print(non_zero_dwt)
    # av_dwt = non_zero_dwt['timestamp'].mean()
    # #av_dwt = datetime.datetime.strptime(av_dwt,"%H:%M:%S")
    # print(av_dwt)
    #
    # """Replace last click duration of each session with the average dwelltime"""
    # dw_t.loc[dw_t['timestamp'] == '00:00:00'] = av_dwt
    # #dw_t['timestamp'].apply(lambda x: av_dwt if x == '00:00:00' else x)
    # dw_t.rename(columns = {"timestamp": "dwelltime"},inplace=True)
    #
    df.reset_index(inplace=True)
    dw_t.reset_index(inplace=True)
    dw_t = pd.DataFrame(dw_t)
    print(dw_t)
    print(df)
    #
    new_df = df.merge(dw_t,on='index',how='inner')
    new_df = pd.DataFrame(new_df)
    new_df.drop(columns='index',inplace=True)
    new_df.to_csv('/home/nick/Desktop/thesis/datasets/cosmetics-shop-data/brand_dataset_dwelltime.csv')