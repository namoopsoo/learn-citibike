
import pandas as pd
import numpy as np
from os import path
import json
import datetime
from utils import calc_speeds
from utils import (distance_between_positions, get_start_time_bucket, 
                which_col_have_nulls)
from sklearn.preprocessing import OneHotEncoder
from annotate_geolocation import annotate_df_with_geoloc

from get_station_geolocation_data import get_station_geoloc_data

import settings as s


def load_data(source_file=s.TRIPS_FILE, num_rows=None):
    ''' Load subscriber data. Non subscriber rows are tossed.
    '''
    df = pd.read_csv(source_file)

    if num_rows:

        if num_rows > df.shape[0]:
            raise Exception, 'source df too small %s for num_rows=%s' % (
                    df.shape[0], num_rows)

        df = df.sample(n=num_rows)

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

def create_annotated_dataset(dataset_name=None,
        dataset_df=None,
        preview_too=True, size=None):
    '''
from pipeline_data import create_annotated_dataset

import pipeline_data as pl
pl.create_annotated_dataset ('201509-citibike-tripdata.csv', size=10000, preview_too=False)

    '''
    if dataset_name:
        df = load_data(path.join(s.DATAS_DIR, dataset_name), num_rows=size)
    elif dataset_df is not None:
        # FIXME >.. if size is None, then dont need to sample,
        #   since other wise the default will be n=1
        min_size = dataset_df.shape[0]
        if size is None:
            size = min_size
        size = min(size, min_size)
        df = dataset_df.sample(n=size)
        df = re_index(df)
    else:
        raise Exception, 'need a source'

    annotated_df = append_travel_stats(df)

    station_dataset = path.join(s.DATAS_DIR, 'stations_geoloc_data.03262016T1349.csv')
    station_df = pd.read_csv(station_dataset)

    next_df = annotate_df_with_geoloc(annotated_df, station_df)

    if not size:
        size = next_df.shape[0]

    timestamp = datetime.datetime.now().strftime('%m%d%YT%H%M')

    dataset_filename = '%s.annotated.%s.%s.csv' % (dataset_name, 
            size, timestamp)
    next_df.to_csv(path.join(s.DATAS_DIR, dataset_filename))

    return dataset_filename


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

        # FIXME ... if shape of df is 1 row, then the assertion fails
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

    station_dataset = path.join(s.DATAS_DIR, 'stations_geoloc_data.03262016T1349.csv')
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
    stations_json_filename = path.join(s.DATAS_DIR, 'start_stations_103115.json')
    stations_json_filename = path.join(s.DATAS_DIR, 'start_stations_060416.json')

    stations_df = get_station_geoloc_data(stations_json_filename)

    # stations_df.to_excel('data/stations_geoloc_030516.xls')
    stations_geoloc_data_filename = path.join(s.DATAS_DIR,
            'stations_geoloc_data.{}.csv'.format(
                datetime.datetime.now().strftime('%m%d%YT%H%M')))
    stations_geoloc_data_filename
    stations_df.to_csv(stations_geoloc_data_filename)

    for i in stations_df.head().index:
        address = stations_df.iloc[i]['station_name']
        print i, address

    
def create_datasets_from_sizes(dataset_source_df, sizes, dry_run=True):
    datasets = []

    for size in sizes:

        if dry_run:
            name = 'd.%s.csv' % size
        else:
            name = create_annotated_dataset(
                    dataset_df=dataset_source_df, 
                    size=size, preview_too=False)

        definition = {'size': size, 'name': name}
        datasets.append(definition)
    return datasets

def make_new_datasets(dataset_source_df, dry_run=True):
    # dataset_source = '201509_10-citibike-tripdata.csv'

    sizes = []
    for i in range(1, 11):
        sizes.append(i*10**5)

    datasets = create_datasets_from_sizes(dataset_source_df,
            sizes, dry_run)
    return datasets
    

def prepare_training_and_holdout_datasets(dataset_source_name):
    '''
    NOTE: this func expects an annotated and de-nulled file.
    '''
    dataset_source = path.join(s.DATAS_DIR, dataset_source_name)
    full_df = load_data(dataset_source)

    holdout_size = 100000
    holdout_df = full_df.sample(n=holdout_size)

    # Do i need to be reindexing here ? 
    holdout_df = re_index(holdout_df)

    # Take out the holdout rows.
    full_df.drop(holdout_df.index, inplace=True, axis=0)
    full_df = re_index(full_df)

    # And make annotated from that holdout.
    annotated_holdout_filename = create_annotated_dataset(
            dataset_df=holdout_df, size=holdout_size)

    # TODO: why didnt this func need resampling when annotating datasets? 
    train_datasets = make_new_datasets(full_df, dry_run=False)

    all_datasets = {
            'holdout_dataset': annotated_holdout_filename,
            'train_datasets': train_datasets}

    return all_datasets


def run_this_08142016():

    dataset_source = path.join(s.DATAS_DIR, '201509_10-citibike-tripdata.csv')
    all_datasets = prepare_training_and_holdout_datasets(dataset_source)

    print all_datasets

def feature_binarization(df, one_hot_encoding):
    '''
    Create a new df which has encoded the columns specified in one_hot_encoding dict.

    EXPECTING: df values are already went through LabelEncode() . i.e. no strings.
    '''
    oh_encoders = {}

    for col in one_hot_encoding:
        col_arr = np.array([[val] for val in df[col]])

        oh_encoders[col] = OneHotEncoder()
        out = oh_encoders[col].fit_transform(col_arr)

        # Give the output columnar df same index, for easy concatenation.
        hot_encoded = pd.DataFrame(out.toarray(), index=df.index)

        # Delete that original col.
        df.drop(col, axis=1, inplace=True)

        # attach new columns to original df...
        df_hot = pd.concat([df, hot_encoded], axis=1)

        df = df_hot

    return df

