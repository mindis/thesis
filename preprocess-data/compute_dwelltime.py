import numpy as np
import pandas as pd


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