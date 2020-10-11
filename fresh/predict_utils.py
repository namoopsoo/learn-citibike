import joblib
import os
import pandas as pd
import numpy as np
from sklearn.datasets import dump_svmlight_file

try:
    import xgboost as xgb
except Exception:
    pass

from io import StringIO

import fresh.utils as fu
import fresh.preproc.v2 as pv2

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

def full_predict(bundle, record):
    X_transformed = X_from_record(bundle, record)
    print(X_transformed)
    dmatrix = xgb.DMatrix(X_transformed)
    
    model = bundle['model_bundle']['bundle']['xgb_model']
    
    y_prob_vec = model.predict(dmatrix)
    predictions = np.argmax(y_prob_vec, axis=1)

    return y_prob_vec, predictions


def full_predict_v2(bundle, record):
    # fork of full_predict with the libsvm hack!
    X_transformed = X_from_record(bundle, record)
    print(X_transformed)

    # dump to a temp location
    temp_loc = '/opt/server/hmmmm.libsvm'
    yblah = np.ones(shape=(1,))
    dump_svmlight_file(X_transformed, yblah, f=temp_loc)
    dmatrix = xgb.DMatrix(f'{temp_loc}?format=libsvm')

    model = bundle['model_bundle']['bundle']['xgb_model']
    
    y_prob_vec = model.predict(dmatrix)
    predictions = np.argmax(y_prob_vec, axis=1)

    return y_prob_vec, predictions


def X_from_record(bundle, record):
    inputdf = pd.DataFrame.from_records([record])
    stationsdf = bundle['stations_bundle']['stationsdf']
    
    X = fu.prepare_data(inputdf, stationsdf, labelled=False)
    
    print(['start_neighborhood', 'gender', 'time_of_day', 'usertype', 'weekday', ])
    print(X)

    X_transformed = pv2.preprocess(
        X=X, 
        neighborhoods=bundle['neighborhoods_bundle']['neighborhoods'], 
        proc_bundle=bundle['proc_bundle']['bundle']['proc_bundle'],
        #workdir=workdir,
        #dataset_name='input'
    )
    return X_transformed


def load_bundle(loc):
    # Need a whole func here, because of this weird OneHotEncoder error,
    # where for some reason, maybe because of a bug in my version of sklearn,
    # I get this weird exception 
    # AttributeError: 'OneHotEncoder' object has no attribute 'drop'
    
    #loc = '/opt/program/artifacts/2020-08-19T144654Z/all_bundle.joblib'
    bundle = joblib.load(loc)

    # fix bundle... 
    bundle['proc_bundle']['bundle']['proc_bundle']['enc'].drop = None
    return bundle


def load_bundle_in_docker():
    bundle_loc = '/opt/program/artifacts/2020-08-19T144654Z/all_bundle.joblib'
    bundle_loc = '/opt/program/artifacts/2020-08-19T144654Z/all_bundle_with_stationsdf.joblib'
    bundle_loc = f'{model_path}/all_bundle_with_stationsdf.joblib'
    print('Loading from bundle_loc', bundle_loc)
    return load_bundle(bundle_loc)


def make_canned_record():
    record = {
     'starttime': '2013-07-01 00:00:00',
     'start station id': 164,
     'start station name': 'E 47 St & 2 Ave',
     'start station latitude': 40.75323098,
     'start station longitude': -73.97032517,
    # unknown
    # 'end station id': 504,
    # 'end station name': '1 Ave & E 15 St',
    # 'end station latitude': 40.73221853,
    # 'end station longitude': -73.98165557,
    # 'stoptime': '2013-07-01 00:10:34',
    # 'tripduration': 634,
     'bikeid': 16950,
     'usertype': 'Customer',
     'birth year': '\\N',
     'gender': 0}
    return record

def hydrate(csvdata):
    # Input expected to have this
    header = ['starttime',
            'start station name',
            'usertype',
            'birth year',
            'gender']

    # Downstream also expects this 
    additional = [
            'bikeid',
            'start station id',
            'start station latitude',
            'start station longitude',
            ]
    df = csv_to_df(csvdata, header)
    return widen_df_with_other_cols(df, header + additional)

def csv_to_df(csvdata, header):
    measured = len(csvdata.split('\n')[0].split(','))
    assert measured == len(header),\
            'len csvdata row ({}) is different than header {}. csvdata: """{}"""'.format(
                    measured, len(header),
                    csvdata)

    return hydrate_inner(header, csvdata)


def hydrate_inner(header, csvdata):
    header_str = ','.join(header)
    full_csvdata = '{}\n{}'.format(header_str, csvdata)
    sio = StringIO(full_csvdata)
    df = pd.read_csv(sio)
    return df


def widen_df_with_other_cols(df, all_columns):
    new_cols = list(set(all_columns)
            - set(df.columns.tolist()))
    for col in new_cols:
        df[col] = np.nan
    return df
