
import pandas as pd
import json
import datetime
from utils import calc_speeds
from utils import (distance_between_positions, get_start_time_bucket, 
                which_col_have_nulls)

from annotate_geolocation import annotate_df_with_geoloc

from get_station_geolocation_data import get_station_geoloc_data

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
    '''Re index w/o gaps in index.

    Reindexing is really important, because without this, future operations,
    on the df, will unknowingly not apply() operations to all rows.
    '''
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

def create_annotated_dataset(dataset_name, preview_too=True, size=None):
    '''
from pipeline_data import create_annotated_dataset

import pipeline_data as pl
pl.create_annotated_dataset ('201509-citibike-tripdata.csv', size=10000, preview_too=False)

    '''
    if preview_too:
        df_mini = load_data('data/%s' % dataset_name, num_rows=10000)
        annotated_df_mini = append_travel_stats(df_mini)
        timestamp = datetime.datetime.now().strftime('%m%d%YT%H%M')
        annotated_df_mini.to_csv(
                'data/%s.annotated.mini.%s.csv' % (dataset_name, 
                    timestamp))

    print 'working on full dataset now...'
    df = load_data('data/%s' % dataset_name, num_rows=size)
    annotated_df = append_travel_stats(df)

    station_dataset = 'data/stations_geoloc_data.03262016T1349.csv'
    station_df = pd.read_csv(station_dataset)

    next_df = annotate_df_with_geoloc(annotated_df, station_df)

    if not size:
        size = next_df.shape[0]

    timestamp = datetime.datetime.now().strftime('%m%d%YT%H%M')
    annotated_df.to_csv(
            'data/%s.annotated.%s.%s.csv' % (dataset_name, 
                size,
                timestamp))


def append_travel_stats(df):
    ''' Annotate regular citibike trip dataframe with derived features.
    '''
    
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

def add_geocoding_station_data(df):
    ''' Enrich input dataframe with station geolocation data.


    '''

    station_dataset = 'data/stations_geoloc_data.03262016T1349.csv'
    station_df = pd.read_csv(station_dataset)

def choose_end_station_label_column(df, label_column):
    '''Annotate citibike df, to use only one label column.
    
    Toss out other end_station columns other than the one chosen.'''
    dump_cols = set(s.ALL_END_STATION_COLS)
    if label_column in dump_cols:
        dump_cols.discard(label_column)
    
    annotated_df = df.drop(dump_cols, axis=1)
    return annotated_df
    
    
def make_geoloc_df():
    ''' Build of stations dataframe.

    File with a list of station intersections like,
    [
    "1 Ave & E 15 St", 
    "1 Ave & E 18 St", 
    "1 Ave & E 30 St", 
    "1 Ave & E 44 St", 
    "1 Ave & E 62 St", 
    "1 Ave & E 68 St", 
    "1 Ave & E 78 St", 
    ...]
    '''
    stations_json_filename = 'data/start_stations_103115.json'
    stations_json_filename = 'data/start_stations_060416.json'

    stations_df = get_station_geoloc_data(stations_json_filename)

    # stations_df.to_excel('data/stations_geoloc_030516.xls')
    stations_geoloc_data_filename = 'data/stations_geoloc_data.{}.csv'.format(
        datetime.datetime.now().strftime('%m%d%YT%H%M'))
    stations_geoloc_data_filename
    stations_df.to_csv(stations_geoloc_data_filename)

    for i in stations_df.head().index:
        address = stations_df.iloc[i]['station_name']
        print i, address

    
