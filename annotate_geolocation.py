



def _make_column_rename_dict(columns, prefix):
    '''Apply prefix and make a dict that can be used with 
    pd.DataFrame.rename() func
    '''
    d = {col: '{}_{}'.format(prefix, col) for col in columns}
    return d


def annotate_df_with_geoloc(df, station_df):

    #
    station_df_columns = [u'station_name', u'postal_code', u'sublocality',
           u'neighborhood', u'state']

    step1_rename_dict = _make_column_rename_dict(station_df_columns, 'start')
    step2_rename_dict = _make_column_rename_dict(station_df_columns, 'end')
    #

    step1_df = pd.merge(left=df, right=station_df, how='inner',
                       left_on=['start station name'],
                        right_on=['station_name'])

    step1_df.rename(columns=step1_rename_dict, inplace=True)

    step2_df = pd.merge(left=step1_df, right=station_df, how='inner',
                       left_on=['end station name'],
                        right_on=['station_name'])

    step2_df.rename(columns=step2_rename_dict, inplace=True)

    # Should toss one of the cols out which were joined on. Also should have initially 
    #   read in df honoring index, so it doesnt get into the data. 

    
    return step2_df
