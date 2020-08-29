import os
import pandas as pd
import joblib
import fresh.map as fm
import fresh.s3utils as fs3
import fresh.predict_utils as fpu

import requests
LOCAL_DIR_SAFE = os.getenv('LOCAL_DIR_SAFE')
LOCAL_URL = 'http://127.0.0.1:8080/invocations'
BUNDLE_LOC_S3 = (
        f"s3://{os.getenv('MODEL_LOC_BUCKET')}/"
        'bikelearn/artifacts/2020-08-19T144654Z/all_bundle_with_stationsdf_except_xgb.joblib'
        )

def entry(event, context):

    # make input into a record
    record = {}
    record = {
     'starttime': '2013-07-01 00:00:00',
     'start station name': 'E 47 St & 2 Ave',
     'usertype': 'Customer',
     'birth year': '1999',
     'gender': 0
     }

    # call sagemaker endpoint
    out = call_sagemaker(record)
    bundle = fetch_bundle()

    probs = map_probabilities(bundle, prob_vec=out['result'][0], k=5)

    blah_get_map(bundle, probs)

    # Translate top 5 results to locations (latlng)
    # and send to google api..


def call_sagemaker(record):


    url = LOCAL_URL
    headers = {'Content-Type': 'text/csv'}
    # tripduration,starttime,stoptime,start station id,start station name,start station latitude,start station longitude,end station id,end station name,end station latitude,end station longitude,bikeid,usertype,birth year,gender\n

    header = ['starttime',
            'start station name',
            'usertype',
            'birth year',
            'gender']


    csvdata = ','.join([str(record[k]) for k in header])

    r = requests.post(url, data=csvdata, headers=headers)
    assert r.status_code/100 == 2, \
            'got this instead, r.status_code ' + str(r.status_code)

    return r.json()


def map_probabilities(bundle, prob_vec, k=5):
    # so lambda downloads bundle from s3 live perhaps
    le = bundle['proc_bundle']['bundle']['proc_bundle']['le']
    classes = le.classes_.shape
    le.classes_ # 54

    top_k = sorted(list(zip(le.classes_, prob_vec)), key=lambda x:x[1], reverse=True)[:k] 
    return top_k


def blah_get_map(bundle, probs):
    stationsdf = bundle['stations_bundle']['stationsdf']
    df = pd.DataFrame(probs, columns=['neighborhood', 'prob'])

    locations = df.merge(
            stationsdf, on='neighborhood'
            ).drop_duplicates(subset='neighborhood')[['latlng', 'neighborhood']
                    ].to_dict(orient='records')
    fm.grab_final_thing(locations)


def fetch_bundle():
    s3uri = BUNDLE_LOC_S3
    local_loc = f'{LOCAL_DIR_SAFE}/blah.joblib'
    fs3.copy_s3_to_local(s3uri, local_loc, force=True)

    return fpu.load_bundle(local_loc)

