import pandas as pd
import os
import unittest

import bikelearn.make_datasets as bm
from bikelearn.models import treefoo
import bikelearn.settings as s
import bikelearn.classify as blc

class DuhPipelineTest(unittest.TestCase):

    def test_treefoo_duh(self):
        fn = 'bikelearn/tests/data/basic-citibike-tripdata.csv'
        df = pd.read_csv(fn)

        from nose.tools import set_trace; set_trace()

        stations_fn = os.path.join(s.DATAS_DIR, 'start_stations_103115.fuller.csv')
        stations_df = pd.read_csv(stations_fn, index_col=0, dtype={'postal_code': str})

        # make simple train/test sets
        datasets = bm.make_dfs(df, stations_df)

        bundle = treefoo.make_tree_foo(
                {'trainset': datasets['train_df'], 'fn': fn},
                {'stations_df': stations_df, 'fn': stations_fn})


        holdout_df = datasets['holdout_df']
        y_preds = blc.run_model_predict(bundle, holdout_df, stations_df)

        pass


