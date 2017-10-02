
import pandas as pd
import numpy as np

def _make_column_rename_dict(columns, prefix):
    '''Apply prefix and make a dict that can be used with 
    pd.DataFrame.rename() func
    '''
    d = {col: '{}_{}'.format(prefix, col) for col in columns}
    return d


def annotate_df_with_geoloc(df, station_df, noisy_nonmatches=False):
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

    # Want to throw exception when rows from df were thrown out because no station
    #   found in the station_df
    missing_stations = (set(np.unique(df['start station name'].values)) 
            - set(np.unique(station_df['station_name'].values))) 

    if noisy_nonmatches:
        assert len(missing_stations) == 0, 'missing stations: %s' % missing_stations

    step1_df = pd.merge(left=df, right=station_df, how='inner',
                       left_on=['start station name'],
                        right_on=['station_name'])


    step1_df.rename(columns=step1_rename_dict, inplace=True)

    step2_df = pd.merge(left=step1_df, right=station_df, how='inner',
                       left_on=['end station name'],
                        right_on=['station_name'])

    step2_df.rename(columns=step2_rename_dict, inplace=True)

    return step2_df


def make_dead_simple_df(annotated_df):

    df = annotated_df.copy()

    out_columns = ['start_sublocality', 'end_sublocality']

    filtered_df = df[out_columns].dropna()


    # Simple encoding
    simple_encoding_map = {'Brooklyn': 1, 'Manhattan': 2, 'Queens': 3}

    for col in out_columns:
        filtered_df[col] = filtered_df[col].apply(lambda x: simple_encoding_map[x])
            
    return filtered_df



