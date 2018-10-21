
import pandas as pd
import numpy as np

import classify
import settings as s

def _make_column_rename_dict(columns, prefix):
    '''Apply prefix and make a dict that can be used with 
    pd.DataFrame.rename() func
    '''
    d = {col: '{}_{}'.format(prefix, col) for col in columns}
    return d

def warn_missing_stations(df, station_df):
    # Want to throw exception when rows from df dont match a station
    #   found in the station_df
    missing_stations = (set(np.unique(df['start station name'].values)) 
            - set(np.unique(station_df['station_name'].values))) 

    if missing_stations:
        print '!missing stations: %s' % missing_stations


def annotate_df_with_geoloc(df, station_df,
        noisy_nonmatches=False):
        #purpose, noisy_nonmatches=False):
    ''' Given a df with both the 'start station name' and 'end station name'
    columns, join with the station_df to also include the geolocation 
    columns, including the start and end postal code, neighborhood and others.

    If noisy_nonmatches, shout when df has stations not found in station_df

    NOTE:
    Before using in training, need to toss one of the cols out which were joined on. 
    Also should have initially read in df honoring index, so it doesnt get into the data. 
    '''

    station_df_columns = [u'station_name', u'postal_code', u'sublocality',
           u'neighborhood', u'state']

    step1_rename_dict = _make_column_rename_dict(station_df_columns, 'start')
    step2_rename_dict = _make_column_rename_dict(station_df_columns, 'end')

    # Want to throw exception when rows from df dont match a station
    #   found in the station_df
    missing_stations = (set(np.unique(df['start station name'].values)) 
            - set(np.unique(station_df['station_name'].values))) 

    if noisy_nonmatches:
        warn_missing_stations(df, station_df)

        if missing_stations:
            print '!missing stations: %s' % missing_stations

    step1_df = pd.merge(left=df, right=station_df, how='left',
                       left_on=['start station name'],
                        right_on=['station_name'])

    # TODO... shouldnt be renaming but just appending...
    step1_df.rename(columns=step1_rename_dict, inplace=True)

    if step1_df[step1_df['end station name'].notnull()].shape[0] > 0:

        step2_df = pd.merge(left=step1_df, right=station_df, how='inner',
                           left_on=['end station name'],
                            right_on=['station_name'])

        step2_df.rename(columns=step2_rename_dict, inplace=True)

        return step2_df
    
    return step1_df


def make_dead_simple_df(annotated_df):
    '''
    Take a simple geo annotated df, and select the output cols for train/test.

    In this case simple geo is just the input/output borough aka sublocality.
    '''

    df = annotated_df.copy()

    out_columns = ['start_sublocality', 'end_sublocality']

    filtered_df = df[out_columns].dropna()


    # Simple encoding
    simple_encoding_map = {'Brooklyn': 1, 'Manhattan': 2, 'Queens': 3}
    for col in out_columns:
        filtered_df[col] = filtered_df[col].apply(lambda x: simple_encoding_map[x])
            
    return filtered_df


def make_medium_simple_df(annotated_df, feature_encoding_dict):
    '''
    Take a geo and time annotated df, and select the output cols for train/test.
    '''

    # output label select....
    # NEW_END_POSTAL_CODE
    # NEW_END_STATE
    # NEW_END_BOROUGH
    # NEW_END_NEIGHBORHOOD

    df = annotated_df.copy()

    out_columns = [
            s.USER_TYPE_COL,
            s.NEW_START_POSTAL_CODE,
            s.NEW_START_BOROUGH, s.NEW_START_NEIGHBORHOOD,
            s.START_DAY, s.START_HOUR,
            s.AGE_COL_NAME, s.GENDER,] + [s.NEW_END_NEIGHBORHOOD]

    for col, dtype in feature_encoding_dict.items():
        df[col] = df[col].astype(dtype)

    dfcopy, label_encoders = classify.build_label_encoders_from_df(
            df, feature_encoding_dict)

    # Apply label encoding...

    # TODO probably need re-indexing?

    return dfcopy, label_encoders

def do_prep(df, feature_encoding_dict):
    for col, dtype in feature_encoding_dict.items():
        if col in df:
            df[col] = df[col].astype(dtype)
    return df

