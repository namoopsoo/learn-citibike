
import pandas as pd
import datetime
from utils import calc_speeds
from utils import (distance_between_positions, get_start_time_bucket, 
                which_col_have_nulls)

import settings as s


def load_data(source_file=s.TRIPS_FILE, num_rows=None):
    ''' Load subscriber data. Non subscriber rows are tossed.
    '''
    df = pd.read_csv(source_file)

    if num_rows:
        df = df[:num_rows]

    df = df[df[s.USER_TYPE_COL] == s.USER_TYPE_SUBSCRIBER]

    df_unnulled = remove_rows_with_nulls(df)
    df_re_index = re_index(df_unnulled)

    return df_re_index

def re_index(df):
    '''Re index w/o gaps in index '''
    df.index = range(df.shape[0])
    return df

def remove_rows_with_nulls(df):
    unnulled = df.dropna()
    return unnulled

def calc_distance_travelled_col(df):
    '''

    TODO... apply in dataframe notation.
            df[START_STATION_LATITUDE_COL],
            df[START_STATION_LONGITUDE_COL],
            df[END_STATION_LATITUDE_COL],
            df[END_STATION_LONGITUDE_COL],

    '''

    values = df.as_matrix(columns=[
        s.START_STATION_LATITUDE_COL,
        s.START_STATION_LONGITUDE_COL,
        s.END_STATION_LATITUDE_COL,
        s.END_STATION_LONGITUDE_COL])

    distances = []

    for row in values:
        distance = distance_between_positions(*row)

        distances.append(distance)

    return distances

def create_annotated_dataset(dataset_name, preview_too=True):
    '''
from pipeline_data import create_annotated_dataset
create_annotated_dataset(dataset_name='201510-citibike-tripdata.csv')

    '''
    if preview_too:
        df_mini = load_data('data/%s' % dataset_name, num_rows=10000)
        annotated_df_mini = append_travel_stats(df_mini)
        timestamp = datetime.datetime.now().strftime('%m%d%YT%H%M')
        annotated_df_mini.to_csv(
                'data/%s.annotated.mini.%s.csv' % (dataset_name, 
                    timestamp))

    print 'working on full dataset now...'
    df = load_data('data/%s' % dataset_name)
    annotated_df = append_travel_stats(df)
    timestamp = datetime.datetime.now().strftime('%m%d%YT%H%M')
    annotated_df.to_csv(
            'data/%s.annotated.%s.csv' % (dataset_name, 
                timestamp))

    print 'done'

def append_travel_stats(df):
    
    recalculate_dict = {
            s.DISTANCE_TRAVELED_COL_NAME: True,
            s.SPEED_COL_NAME: True, 
            s.AGE_COL_NAME: True,
            s.START_TIME_BUCKET: True,
            }

    assert not which_col_have_nulls(df)

    if recalculate_dict[s.DISTANCE_TRAVELED_COL_NAME]:
        dist_travelled = calc_distance_travelled_col(df)
        df[s.DISTANCE_TRAVELED_COL_NAME] = pd.Series(dist_travelled)

        assert not which_col_have_nulls(df)

    if recalculate_dict[s.SPEED_COL_NAME]:
        travel_speeds = calc_speeds(df)
        df[s.SPEED_COL_NAME] = pd.Series(travel_speeds)

        assert not which_col_have_nulls(df)

    if recalculate_dict[s.AGE_COL_NAME]:
        df[s.AGE_COL_NAME] = 2015 - df[s.BIRTH_YEAR_COL]

        assert not which_col_have_nulls(df)

    if recalculate_dict[s.START_TIME_BUCKET]:
        time_buckets = calculate_start_time_buckets(df)
        df[s.START_TIME_BUCKET] = pd.Series(time_buckets)

        assert not which_col_have_nulls(df)

    return df

def calculate_start_time_buckets(df):

    values = df.as_matrix(columns=[
        s.START_TIME])

    start_time_buckets = []

    for row in values:
        buck = get_start_time_bucket(row[0])
        start_time_buckets.append(buck)

    
    return start_time_buckets


