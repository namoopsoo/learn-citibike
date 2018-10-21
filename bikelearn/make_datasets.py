import sys
import datetime
import pytz
import pandas as pd

import bikelearn.classify as cl


import os
import pandas as pd
import pipeline_data as pl
import annotate_geolocation as annotate_geo
import settings as s

def make_dfs(indf, stations_df):



    train_df, holdout_df = cl.simple_split(indf)

    return {'train_df': train_df, 
            'holdout_df': holdout_df,}

    # hmmm why was i doing this here ... hmmm.. dont think this belongs here.
    next_df = annotate_geo.annotate_df_with_geoloc(train_df, stations_df, noisy_nonmatches=False)

    and_age_df = pl.annotate_age(next_df)
    more_df = pl.annotate_time_features(and_age_df)

    simpledf, label_encoders = annotate_geo.make_medium_simple_df(more_df)

    # train_df, holdout_df = cl.simple_split(simpledf)
    return {'train_df': train_df, 
            'holdout_df': holdout_df,}
            #'label_encoders': label_encoders}

    
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

