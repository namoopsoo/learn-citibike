import pandas as pd
import os

import bikelearn.settings as s
from bikelearn.models import treefoo
import bikelearn.make_datasets as bm

def make_basic_minimal_model():
    fn = 'bikelearn/tests/data/basic-citibike-tripdata.csv'
    df = pd.read_csv(fn)

    stations_fn = os.path.join(s.DATAS_DIR, 'start_stations_103115.fuller.csv')
    stations_df = pd.read_csv(stations_fn, index_col=0, dtype={'postal_code': str})

    # make simple train/test sets
    datasets = bm.make_dfs(df, stations_df)

    bundle = treefoo.make_tree_foo(
            {'trainset': datasets['train_df'], 'fn': fn},
            {'stations_df': stations_df, 'fn': stations_fn})

    assert s.USER_TYPE_COL in bundle['label_encoders']

    return bundle, datasets, stations_df


def get_basic_proportion_correct(y_test, y_predictions):
    zipped = zip(y_test, y_predictions)
    correct = len([[x,y] for x,y in zipped if x == y])
    proportion_correct = 1.0*correct/y_test.shape[0]
    return proportion_correct

