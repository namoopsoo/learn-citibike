import joblib
import pandas as pd
import numpy as np
import xgboost as xgb

import fresh.utils as fu
import fresh.preproc.v2 as pv2

def full_predict(bundle, record):
    inputdf = pd.DataFrame.from_records([record])
    
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
    print(X_transformed)
    dtrain = xgb.DMatrix(X_transformed)
    
    model = bundle['model_bundle']['bundle']['xgb_model']
    
    y_prob_vec = model.predict(dtrain)
    predictions = np.argmax(y_prob_vec, axis=1)

    return y_prob_vec, predictions

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



