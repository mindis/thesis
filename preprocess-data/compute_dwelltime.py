import numpy as np
import pandas as pd
import datetime


userkey = 'user_id' # or 'user_session'
itemkey = 'product_id'
timekey = 'event_time'
brandkey = 'brand'

def compute_dwell_time(df,userkey,itemkey,timekey):
    times_t = np.roll(df[timekey], -1)  # Take time row
    times_dt = df[timekey]  # Copy, then displace by one

    diffs = np.subtract(times_t, times_dt)  # Take the pairwise difference

    length = len(df[itemkey])

    # cummulative offset start for each session
    offset_sessions = np.zeros(df[userkey].nunique() + 1, dtype=np.int32)
    offset_sessions[1:] = df.groupby(userkey).size().cumsum()

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

def get_session_duration_arr(data,userkey,timekey):
    """Compute session duration for each session"""
    data.event_time = pd.to_datetime(data.event_time)
    df = data.groupby(userkey)[timekey].agg(
        lambda x: max(x) - min(x)).to_frame().rename(columns={timekey: 'Duration'})
    df = pd.DataFrame(df)
    return df

if __name__ == "__main__":

    #PATH = '/home/nick/Desktop/thesis/datasets/cosmetics-shop-data/brand_dataset.csv'
    PATH = '/home/nick/Desktop/thesis/datasets/pharmacy-data/cleaned_users_products.csv'
    userkey = 'user_id'
    itemkey = 'product_id'
    timekey = 'timestamp'

    df = pd.read_csv(PATH)
    df.drop(columns=['Unnamed: 0'],inplace=True)
    print(df)
    df.sort_values(by=[userkey,timekey],inplace=True)
    df[timekey] = pd.to_datetime(df[timekey])
    df[timekey]=df[timekey].dt.tz_localize(None)
    print(df)
    # df[timekey] = datetime.datetime.strptime()
    # print(df)
    """Compute dwelltime"""
    dw_t = compute_dwell_time(df,userkey,itemkey,timekey)
    dw_t = pd.DataFrame(dw_t)

    final_df = df

    """The last clickstream info of every session has invalid dwelltime due to the substraction of 2 different time periods
    So we will clean these outliers and will take the average dwelltime of the session instead"""

    """Calculate the average dwelltime of the records, ignoring outliers"""
    cleaned_dwt = dw_t[(dw_t[timekey] != '0 days 00:00:00') & (dw_t[timekey] < '0 days 01:00:00')]
    #cleaned_dwt = dw_t[dw_t[timekey] != '0 days 00:00:00']
    cleaned_dwt.rename(columns = {"timestamp": "dwelltime"},inplace=True)
    print(cleaned_dwt)

    av_dwt = cleaned_dwt['dwelltime'].mean()
    print(av_dwt)


    """Replace last click duration of each session with the average dwelltime"""
    dw_t.loc[(dw_t[timekey] == '0 days 00:00:00') ^ (dw_t[timekey] >= '0 days 01:00:00')] = av_dwt
    #dw_t['timestamp'].apply(lambda x: av_dwt if x == '00:00:00' else x)
    dw_t.rename(columns = {"timestamp": "dwelltime"},inplace=True)
    # print(type(dw_t['dwelltime'][2]))
    print(dw_t)
    #print(final_df)
    final_df['dwelltime'] = dw_t['dwelltime']
    final_df = pd.DataFrame(final_df)
    dwelltime_freqs = pd.DataFrame(final_df['dwelltime'].value_counts())

    #Extract to csv
    final_df.to_csv('/home/nick/Desktop/thesis/datasets/pharmacy-data/user_products_dwelltime.csv')

    """Calculate mean dwelltime for each session"""

    # mean_dwells = pd.DataFrame(merged_df1[['user_id','dwelltime']].groupby(userkey).mean(numeric_only=False))
    # print(mean_dwells)
    # # mean_dwells.reset_index(inplace=True)
    # # print(mean_dwells)
    # print(mean_dwells.loc[merged_df1[userkey][1],'dwelltime'])
    #
    #
    # merged_df2 = pd.concat([df, cleaned_dwt], axis=1)
    # #print(merged_df2)
    # print(merged_df2['dwelltime'][2])


    # for i in range(len(merged_df2)):
    #     if pd.isna(merged_df2['dwelltime'][i]):
    #         merged_df2['dwelltime'][i] = 1
    #         #mean_dwells.loc[merged_df2[userkey][i], 'dwelltime']
    # print(merged_df2)







