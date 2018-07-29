import sys
import datetime
import pytz
import pandas as pd

import classify as cl


import os
import pandas as pd
import pipeline_data as pl
import annotate_geolocation as annotate_geo
import settings as s

def make_dfs(indf):
    stations_df_filename = os.path.join(s.DATAS_DIR, 'start_stations_103115.fuller.csv')
    stations_df = pd.read_csv(stations_df_filename, index_col=0, dtype={'postal_code': str})


    next_df = annotate_geo.annotate_df_with_geoloc(indf, stations_df, noisy_nonmatches=False)

    and_age_df = pl.annotate_age(next_df)
    more_df = pl.annotate_time_features(and_age_df)

    simpledf = annotate_geo.make_medium_simple_df(more_df)

    train_df, holdout_df = classify.simple_split(simpledf)
    
def get_timestamp():
    now = datetime.datetime.utcnow().replace(tzinfo=pytz.UTC)
    return now.strftime('%Y-%m-%dT%H%M%S')

def do(source_file, out_dir):
    print source_file, out_dir

    df = pd.read_csv(source_file)
    import ipdb ; ipdb.set_trace();

    train_df, test_df = cl.simple_split(df)
    # put the y in the first col... when saving . 
    ts = get_timestamp()
    pass
    [df.to_csv(os.path.join(out_dir,
        '{}.{}.csv'.format(fn, ts)))
        for df, fn in [[train_df, 'train'],
            [test_df, 'test']]
        ]



if __name__ == '__main__':

    source_file = sys.argv[1]
    out_dir = sys.argv[2]

    do(source_file, out_dir)

