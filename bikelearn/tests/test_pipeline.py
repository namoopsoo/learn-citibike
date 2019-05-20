import pandas as pd
import numpy as np
import requests
import os
import unittest

import bikelearn.make_datasets as bm
from bikelearn.models import treefoo
import bikelearn.settings as s
import bikelearn.classify as blc
import bikelearn.tests.utils as bltu
from bikelearn import pipeline_data as pl


class DuhPipelineTest(unittest.TestCase):

    def test_treefoo_duh(self):
        bundle, datasets, stations_df = bltu.make_basic_minimal_model()

        holdout_df = datasets['holdout_df']


        y_predictions, y_test, metrics = blc.run_model_predict(
                bundle, holdout_df, stations_df, labeled=True)

        proportion_correct_labeled_true = bltu.get_basic_proportion_correct(
                y_test, y_predictions)

        # Again but unlabeled now.

        contracted_df = blc.contract_df(holdout_df)

        widened_df = blc.widen_df_with_other_cols(contracted_df, s.ALL_COLUMNS)

        y_predictions_from_widened, _, _ = blc.run_model_predict(
                bundle, widened_df, stations_df, labeled=False)

        proportion_correct_labeled_false = bltu.get_basic_proportion_correct(
                y_test, y_predictions_from_widened)

        # and assert, evaluation should be the same..
        assert proportion_correct_labeled_false  == proportion_correct_labeled_true
        pass


    def test_treefoo_with_pure_input_data(self):
        # csvdata = 'starttime,start station name,usertype,birth year,gender\n10/1/2015 00:00:02,W 26 St & 10 Ave,Subscriber,1973,1\n10/1/2015 00:00:02,E 39 St & 2 Ave,Subscriber,1990,1'

        csvdata = '10/1/2015 00:00:02,W 26 St & 10 Ave,Subscriber,1973,1\n10/1/2015 00:00:02,E 39 St & 2 Ave,Subscriber,1990,1'


        df = blc.hydrate_csv_to_df(csvdata)

        bundle, datasets, stations_df = bltu.make_basic_minimal_model()

        widened_df = blc.widen_df_with_other_cols(df, s.ALL_COLUMNS)

        y_predictions, _, _ = blc.run_model_predict(
                bundle, widened_df, stations_df, labeled=False)

        pass


class IntegrationLocalTest(unittest.TestCase):
    def setUp(self):
        url = 'http://127.0.0.1:8080/ping'
        assert bltu.ping_service(url), 'local docker is down. Cant test.'

    def test_too_many_columns_400(self):
        url = 'http://127.0.0.1:8080/invocations'
        headers = {'Content-Type': 'text/csv'}
        # tripduration,starttime,stoptime,start station id,start station name,start station latitude,start station longitude,end station id,end station name,end station latitude,end station longitude,bikeid,usertype,birth year,gender\n
        data = '171,10/1/2015 00:00:02,10/1/2015 00:02:54,388,W 26 St & 10 Ave,40.749717753,-74.002950346,494,W 26 St & 8 Ave,40.74734825,-73.99723551,24302,Subscriber,1973.0,1\n593,10/1/2015 00:00:02,10/1/2015 00:09:55,518,E 39 St & 2 Ave,40.74780373,-73.97344190000001,438,St Marks Pl & 1 Ave,40.72779126,-73.98564945,19904,Subscriber,1990.0,1\n'

        r = requests.post(url, data=data, headers=headers)
        assert r.status_code/100 != 2, \
                'got this instead, r.status_code ' + str(r.status_code)


    def test_just_inputs(self):
        url = 'http://127.0.0.1:8080/invocations'
        headers = {'Content-Type': 'text/csv'}
        data = '10/1/2015 00:00:02,W 26 St & 10 Ave,Subscriber,1973,1\n10/1/2015 00:00:02,E 39 St & 2 Ave,Subscriber,1990,1'

        r = requests.post(url, data=data, headers=headers)
        assert r.status_code/100 == 2

        pass

    def test_with_garbage_station(self):

        url = 'http://127.0.0.1:8080/invocations'
        headers = {'Content-Type': 'text/csv'}
        data_vec = [
                '10/1/2015 00:00:02,Foo Rd & Fake St.,Subscriber,1973,1',
                '10/1/2015 00:00:02,E 39 St & 2 Ave,Subscriber,1990,1']
        data = '\n'.join(data_vec)

        r = requests.post(url, data=data, headers=headers)
        assert r.status_code/100 == 2


    def test_just_inputs_w_o_header(self):

        url = 'http://127.0.0.1:8080/invocations'
        headers = {'Content-Type': 'text/csv'}
        data_vec = ['10/1/2015 00:00:02,W 26 St & 10 Ave,Subscriber,1973,1',
                '10/1/2015 00:00:02,E 39 St & 2 Ave,Subscriber,1990,1']
        data = '\n'.join(data_vec)

        r = requests.post(url, data=data, headers=headers)
        assert r.status_code/100 == 2
    
    def test_single_row_though(self):
        url = 'http://127.0.0.1:8080/invocations'
        headers = {'Content-Type': 'text/csv'}
        data_vec = [
                '10/1/2015 00:00:02,E 39 St & 2 Ave,Subscriber,1990,1']
        data = '\n'.join(data_vec)

        r = requests.post(url, data=data, headers=headers)
        assert r.status_code/100 == 2
    


class TestNanLabelEncIssue(unittest.TestCase):
    def test_assertion_happens_for_nan(self):
        _, datasets, stations_df = bltu.make_basic_minimal_model()
        df = pd.DataFrame({s.NEW_END_NEIGHBORHOOD: ['foo', 'nan']})

        feature_encoding_dict = {
                s.NEW_END_NEIGHBORHOOD: str}

        asserted = False
        try:
            _, label_encoders = pl.make_simple_df_from_raw(
                    df, stations_df,
                    feature_encoding_dict)
        except Exception:
            asserted = True


        assert asserted


class TestWidenDfForPredictPipeline(unittest.TestCase):
    def test_basic(self):
        csvdata = '10/1/2015 00:00:02,W 26 St & 10 Ave,Subscriber,1973,1\n10/1/2015 00:00:02,E 39 St & 2 Ave,Subscriber,1990,1'

        df = blc.hydrate_and_widen(csvdata)
        bundle, datasets, stations_df = bltu.make_basic_minimal_model()
        y_predictions, _, _ = blc.run_model_predict(
                bundle, df, stations_df, labeled=False)
        pass

    def test_more_cols_hmm(self):
        csvdata = '171,10/1/2015 00:00:02,10/1/2015 00:02:54,388,W 26 St & 10 Ave,40.749717753,-74.002950346,494,W 26 St & 8 Ave,40.74734825,-73.99723551,24302,Subscriber,1973.0,1\n593,10/1/2015 00:00:02,10/1/2015 00:09:55,518,E 39 St & 2 Ave,40.74780373,-73.97344190000001,438,St Marks Pl & 1 Ave,40.72779126,-73.98564945,19904,Subscriber,1990.0,1\n'

        assert_raised = False
        try:
            df = blc.hydrate_and_widen(csvdata)
        except AssertionError:
            assert_raised = True
        assert assert_raised



class TestAgeFeature(unittest.TestCase):
    def test_basic(self):
        df = pd.DataFrame({s.BIRTH_YEAR_COL: ['1995.0', 1995, '1996', np.nan]})

        df2 = pl.annotate_age(df)
        out = df2[s.AGE_COL_NAME].tolist()
        assert out[:-1] == [20, 20, 19]
        assert pd.isnull(out[-1])

class TestForestShowDecisionPath(unittest.TestCase):
    def test_basic(self):
        from nose.tools import set_trace; set_trace()

        csvdata = '10/1/2015 00:00:02,E 39 St & 2 Ave,Subscriber,1990,1'

        df = blc.hydrate_and_widen(csvdata)

        bundle, datasets, stations_df = bltu.make_basic_minimal_model()

        _, X_test = blc.df_to_np_for_clf(bundle, df, stations_df, labeled=False)
        clf = bundle['clf']

        path = clf.decision_path(X_test)

#y_predictions, _, _ = blc.run_model_predict(
#        bundle, df, stations_df, labeled=False)




class IntegrationLocalValidationCsvTest(unittest.TestCase):
    def setUp(self):

        # Hmm... do a docker run here and then kill  at the tearDown? 
        url = 'http://127.0.0.1:8080/ping'
        assert bltu.ping_service(url), 'local docker is down. Cant test.'

    def test_short(self):
        url = 'http://127.0.0.1:8080/invocations'
        headers = {'Content-Type': 'text/csv'}
        data = (
        'tripduration,starttime,stoptime,start station id,start station name,start station latitude,start station longitude,end station id,end station name,end station latitude,end station longitude,bikeid,usertype,birth year,gender\n'
        '171,10/1/2015 00:00:02,10/1/2015 00:02:54,388,W 26 St & 10 Ave,40.749717753,-74.002950346,494,W 26 St & 8 Ave,40.74734825,-73.99723551,24302,Subscriber,1973.0,1\n'
        '593,10/1/2015 00:00:02,10/1/2015 00:09:55,518,E 39 St & 2 Ave,40.74780373,-73.97344190000001,438,St Marks Pl & 1 Ave,40.72779126,-73.98564945,19904,Subscriber,1990.0,1\n'
        )

        from nose.tools import set_trace; set_trace()

        r = requests.post(url, data=data, headers=headers)
        assert r.status_code/100 == 2, \
                'got this instead, r.status_code ' + str(r.status_code)

        pass




