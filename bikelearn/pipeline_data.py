import sys
import pandas as pd
import numpy as np
import os
from os import path
import json
import datetime
import pytz
import utils

from sklearn.preprocessing import OneHotEncoder
import annotate_geolocation as annotate_geo
from get_station_geolocation_data import get_station_geoloc_data
import dfutils as dfu
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

    df_unnulled = dfu.remove_rows_with_nulls(df)
    df_re_index = dfu.re_index(df_unnulled)

    return df_re_index


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
        distance = utils.distance_between_positions(*row)

        distances.append(distance)

    return distances


def make_timestamp():
    return datetime.datetime.utcnow().replace(tzinfo=pytz.UTC).strftime('%Y-%m-%dT%H%M%SZUTC')


def create_annotated_dataset(identifier=None, dataset_name=None,
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
        df = dfu.re_index(df)
    else:
        raise Exception, 'need a source'

    annotated_df = append_travel_stats(df)

    station_dataset = path.join(s.DATAS_DIR, 'stations_geoloc_data.03262016T1349.csv')
    station_df = pd.read_csv(station_dataset)

    next_df = annotate_geo.annotate_df_with_geoloc(annotated_df, station_df)

    if not size:
        size = next_df.shape[0]

    timestamp = datetime.datetime.now().strftime('%m%d%YT%H%M')

    dataset_filename = '%s.annotated.%s.%s.%s.csv' % (dataset_name, 
            identifier, size, timestamp)
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

    assert not utils.which_col_have_nulls(df)

    if recalculate_dict[s.DISTANCE_TRAVELED_COL_NAME]:
        dist_travelled = calc_distance_travelled_col(df)
        df[s.DISTANCE_TRAVELED_COL_NAME] = pd.Series(dist_travelled)

        # FIXME ... if shape of df is 1 row, then the assertion fails
        assert not utils.which_col_have_nulls(df)

    if recalculate_dict[s.SPEED_COL_NAME]:
        travel_speeds = utils.calc_speeds(df)
        df[s.SPEED_COL_NAME] = pd.Series(travel_speeds)

        assert not utils.which_col_have_nulls(df)

    if recalculate_dict[s.AGE_COL_NAME]:
        df[s.AGE_COL_NAME] = 2015 - df[s.BIRTH_YEAR_COL]

        assert not utils.which_col_have_nulls(df)

    if recalculate_dict[s.START_TIME_BUCKET]:
        time_buckets = calculate_start_time_buckets(df)
        df[s.START_TIME_BUCKET] = pd.Series(time_buckets)

        assert not utils.which_col_have_nulls(df)

    return df


def annotate_age(df):
    df[s.AGE_COL_NAME] = df[s.BIRTH_YEAR_COL].map(
            lambda x: 2015 - int(float(x))
            if not pd.isnull(x) else x)

    return df


def annotate_time_features(df):
    df[s.START_DAY] = df[s.START_TIME].apply(utils.get_start_day)
    df[s.START_HOUR] = df[s.START_TIME].apply(utils.get_start_hour)
    return df


def calculate_start_time_buckets(df):
    values = df.as_matrix(columns=[
        s.START_TIME])

    start_time_buckets = []

    for row in values:
        buck = utils.get_start_time_bucket(row[0])
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
                    identifier='test',
                    dataset_df=dataset_source_df, 
                    size=size, preview_too=False)

        definition = {'size': size, 'name': name}
        datasets.append(definition)
    return datasets

def make_new_datasets(dataset_source_df, dry_run=True):
    # dataset_source = '201509_10-citibike-tripdata.csv'

    sizes = []
    for i in range(1, 12, 2):
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
    holdout_df = dfu.re_index(holdout_df)

    # Take out the holdout rows.
    full_df.drop(holdout_df.index, inplace=True, axis=0)
    full_df = dfu.re_index(full_df)

    # And make annotated from that holdout.
    annotated_holdout_filename = create_annotated_dataset(
            identifier='holdout',
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


def make_one_hot_encoders(df, one_hot_encoding):
    oh_encoders = {}
    for col in one_hot_encoding:
        col_arr = np.array([[val] for val in df[col]])

        oh_encoders[col] = OneHotEncoder()
        oh_encoders[col].fit(col_arr)

    return oh_encoders


def feature_binarization(df, oh_encoders):
    '''
    Create a new df which has encoded the columns specified in one_hot_encoding dict.

    EXPECTING: df values are already went through LabelEncode() . i.e. no strings.
    '''
    # oh_encoders = {}
    # TODO... new col names with base of original to support more than one such encoding.

    for col in oh_encoders:
        col_arr = np.array([[val] for val in df[col]])

        # oh_encoders[col] = OneHotEncoder()
        out = oh_encoders[col].transform(col_arr)

        # Give the output columnar df same index, for easy concatenation.
        num_new_cols = out.shape[1]
        hot_encoded = pd.DataFrame(out.toarray(), index=df.index,
                columns=['{}_{}'.format(col, i)
                    for i in range(num_new_cols)])

        # Delete that original col.
        df.drop(col, axis=1, inplace=True)

        # attach new columns to original df...
        df_hot = pd.concat([df, hot_encoded], axis=1)

        df = df_hot

    return df


def make_simple_df_from_raw(indf, stations_df, feature_encoding_dict):

    next_df = annotate_geo.annotate_df_with_geoloc(indf, stations_df, noisy_nonmatches=True)

    and_age_df = annotate_age(next_df)
    more_df = annotate_time_features(and_age_df)
    simpledf, label_encoders = annotate_geo.make_medium_simple_df(
            more_df, feature_encoding_dict)

    assert not any(['nan' in le.classes_ for le in label_encoders.values()]), \
            {feature_name: le.classes_ for (feature_name, le) in label_encoders.items()
                    if 'nan' in le.classes_}

    return simpledf, label_encoders


def prepare_test_data_for_predict(indf, stations_df,
        feature_encoding_dict, labeled):
    out_columns = [s.NEW_START_POSTAL_CODE,
             s.NEW_START_BOROUGH, s.NEW_START_NEIGHBORHOOD,
             s.START_DAY, s.START_HOUR,
             s.AGE_COL_NAME, s.GENDER,
             s.USER_TYPE_COL]

    if labeled:
        out_columns += [s.NEW_END_NEIGHBORHOOD]


    next_df = annotate_geo.annotate_df_with_geoloc(indf, stations_df, noisy_nonmatches=False)

    and_age_df = annotate_age(next_df)
    more_df = annotate_time_features(and_age_df)

    simpledf = annotate_geo.do_prep(more_df,
            feature_encoding_dict)

    return simpledf[out_columns]


def ship_training_data_to_s3():
    pass


