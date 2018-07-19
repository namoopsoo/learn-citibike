


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

