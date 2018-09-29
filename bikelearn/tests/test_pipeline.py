import pandas as pd
import requests
import os
import unittest

import bikelearn.make_datasets as bm
from bikelearn.models import treefoo
import bikelearn.settings as s
import bikelearn.classify as blc


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


class DuhPipelineTest(unittest.TestCase):

    def test_treefoo_duh(self):
        bundle, datasets, stations_df = make_basic_minimal_model()

        holdout_df = datasets['holdout_df']

        y_predictions, y_test = blc.run_model_predict(
                bundle, holdout_df, stations_df, labeled=True)

        pass

    def test_treefoo_with_pure_input_data(self):
        csvdata = 'starttime,start station name,usertype,birth year,gender\n10/1/2015 00:00:02,W 26 St & 10 Ave,Subscriber,1973,1\n10/1/2015 00:00:02,E 39 St & 2 Ave,Subscriber,1990,1'

        df = blc.hydrate_csv_to_df(csvdata)
        # minimal_cols = 
        # from nose.tools import set_trace; set_trace()

        bundle, datasets, stations_df = make_basic_minimal_model()

        all_columns = ['tripduration', 'starttime', 'stoptime', 'start station id', 'start station name', 'start station latitude', 'start station longitude', 'end station id', 'end station name', 'end station latitude', 'end station longitude', 'bikeid', 'usertype', 'birth year', 'gender']

        widened_df = blc.widen_df_with_other_cols(df, all_columns)


        y_predictions, _ = blc.run_model_predict(
                bundle, widened_df, stations_df, labeled=False)

        pass


class IntegrationLocalTest(unittest.TestCase):

    def test_foo(self):
        url = 'http://127.0.0.1:8080/invocations'
        headers = {'Content-Type': 'text/csv'}
        data = 'tripduration,starttime,stoptime,start station id,start station name,start station latitude,start station longitude,end station id,end station name,end station latitude,end station longitude,bikeid,usertype,birth year,gender\n171,10/1/2015 00:00:02,10/1/2015 00:02:54,388,W 26 St & 10 Ave,40.749717753,-74.002950346,494,W 26 St & 8 Ave,40.74734825,-73.99723551,24302,Subscriber,1973.0,1\n593,10/1/2015 00:00:02,10/1/2015 00:09:55,518,E 39 St & 2 Ave,40.74780373,-73.97344190000001,438,St Marks Pl & 1 Ave,40.72779126,-73.98564945,19904,Subscriber,1990.0,1\n'
        data = 'tripduration,starttime,stoptime,start station id,start station name,start station latitude,start station longitude,end station id,end station name,end station latitude,end station longitude,bikeid,usertype,birth year,gender\n171,10/1/2015 00:00:02,10/1/2015 00:02:54,388,W 26 St & 10 Ave,40.749717753,-74.002950346,494,W 26 St & 8 Ave,40.74734825,-73.99723551,24302,Subscriber,1973.0,1\n593,10/1/2015 00:00:02,10/1/2015 00:09:55,518,E 39 St & 2 Ave,40.74780373,-73.97344190000001,438,St Marks Pl & 1 Ave,40.72779126,-73.98564945,19904,Subscriber,1990.0,1\n'

        r = requests.post(url, data=data,headers=headers)
        from nose.tools import set_trace; set_trace()
        assert r.status_code/100 == 2

        pass

    def test_just_inputs(self):

        url = 'http://127.0.0.1:8080/invocations'
        headers = {'Content-Type': 'text/csv'}
        data = 'starttime,start station name,usertype,birth year,gender\n10/1/2015 00:00:02,W 26 St & 10 Ave,Subscriber,1973,1\n10/1/2015 00:00:02,E 39 St & 2 Ave,Subscriber,1990,1'

        r = requests.post(url, data=data,headers=headers)
        from nose.tools import set_trace; set_trace()
        assert r.status_code/100 == 2

        pass

