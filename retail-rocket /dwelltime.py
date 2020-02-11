import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from build_data_rnn import clean_dataset
from data_exploration import events_explored


def get_session_duration_arr(df):
    # Compute session duration for each visitor
    df.timestamp = pd.to_datetime(df.timestamp)
    data = df.groupby('visitorid')['timestamp'].agg(
        lambda x: max(x) - min(x)).to_frame().rename(columns={'Timestamp': 'Duration'})
    data = pd.DataFrame(data)

    return data

def compute_dwell_time(df):
    times_t = np.roll(df['timestamp'], -1)  # Take time row
    times_dt = df['timestamp']  # Copy, then displace by one

    diffs = np.subtract(times_t, times_dt)  # Take the pairwise difference

    length = len(df['itemid'])

    # cummulative offset start for each session
    offset_sessions = np.zeros(df['visitorid'].nunique() + 1, dtype=np.int32)
    offset_sessions[1:] = df.groupby('visitorid').size().cumsum()

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
    EVENTS_PATH = '/home/nick/Desktop/thesis/datasets/retail-rocket/events.csv'

    events, visitors = events_explored(EVENTS_PATH)

    df = clean_dataset(events)

    ses_duration = get_session_duration_arr(df)
    print(ses_duration)

    #dw_t = compute_dwell_time(df)
    #print(dw_t)

    # new_df = join_dwell_reps(df, dw_t, threshold=200000)
    #
    # print(new_df)




