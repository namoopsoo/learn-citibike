import pandas as pd
import requests
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

        stations_fn = os.path.join(s.DATAS_DIR, 'start_stations_103115.fuller.csv')
        stations_df = pd.read_csv(stations_fn, index_col=0, dtype={'postal_code': str})

        # make simple train/test sets
        datasets = bm.make_dfs(df, stations_df)

        bundle = treefoo.make_tree_foo(
                {'trainset': datasets['train_df'], 'fn': fn},
                {'stations_df': stations_df, 'fn': stations_fn})

        assert s.USER_TYPE_COL in bundle['label_encoders']


        holdout_df = datasets['holdout_df']
        y_predictions, y_test = blc.run_model_predict(
                bundle, holdout_df, stations_df)

        pass


class IntegrationLocalTest(unittest.TestCase):

    def test_foo(self):
        url = 'http://127.0.0.1:8000/invocations'
        data = ',tripduration,starttime,stoptime,start station id,start station name,start station latitude,start station longitude,end station id,end station name,end station latitude,end station longitude,bikeid,usertype,birth year,gender\n0,171,10/1/2015 00:00:02,10/1/2015 00:02:54,388,W 26 St & 10 Ave,40.749717753,-74.002950346,494,W 26 St & 8 Ave,40.74734825,-73.99723551,24302,Subscriber,1973.0,1\n1,593,10/1/2015 00:00:02,10/1/2015 00:09:55,518,E 39 St & 2 Ave,40.74780373,-73.97344190000001,438,St Marks Pl & 1 Ave,40.72779126,-73.98564945,19904,Subscriber,1990.0,1\n'

        r = requests.post(url, data=data)

    
        pass
        pass
        pass

