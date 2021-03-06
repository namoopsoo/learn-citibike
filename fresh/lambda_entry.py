import os
import json
import pickle
import botocore
import boto3
import pandas as pd
import fresh.map as fm
import fresh.s3utils as fs3
import fresh.predict_utils as fpu
import fresh.utils as fu

import requests
LOCAL_DIR_SAFE = os.getenv('LOCAL_DIR_SAFE')
LOCAL_URL = 'http://127.0.0.1:8080/invocations'
BUNDLE_LOC_S3 = (
        f"s3://{os.getenv('MODEL_LOC_BUCKET')}/"
        'bikelearn/artifacts/2020-08-19T144654Z/all_bundle_with_stationsdf_except_xgb.pkl'
        )

def entry(event, context):
    '''
    Example of proper event:
    {
      "params": {
        "path": {},
        "querystring": {
          "birth_year": "1973",
          "rider_gender": "2",
          "rider_type": "Subscriber",
          "start_station": "W+26+St+&+10+Ave",
          "start_time": "2020-10-19+13:56:00"
        }
      }
    }
    # And after quick mappin, a record is created...      
    record = {
     'starttime': '2013-07-01 00:00:00',
     'start station name': 'E 47 St & 2 Ave',
     'usertype': 'Customer',
     'birth year': '1999',
     'gender': 0
     }
    '''
    print('DEBUG', event)
    # make input into a record
    input_record = event['params']['querystring']

    # {'birth_year': '', 'rider_gender': '', 'rider_type': '', 'start_station': '', 'start_time': ''}
    record = {
     'starttime': input_record['start_time'].replace('+', ' '), #'2013-07-01 00:00:00',   # apigateway .. 01/07/2013 00:00:00
     'start station name': input_record['start_station'].replace('+', ' '), #'E 47 St & 2 Ave',
     'usertype': input_record['rider_type'], # 'Customer',
     'birth year': input_record['birth_year'], #'1999',
     'gender': input_record['rider_gender'], #0
     }

    print('DEBUG, new record', record)

    # call sagemaker endpoint
    bundle = fetch_bundle()
    start_location = get_start_location(record, bundle)
    print('DEBUG, start_location', start_location)
    if start_location is None:
        raise Exception('unknown start station')

    out = call_sagemaker(record)
    print('DEBUG, call_sagemaker out', out)

    probs = map_probabilities(bundle,
                              prob_vec=[round(x, 2)
                                        for x in out['result'][0]], k=9)
    out = blah_get_map(bundle, probs, start_location=start_location)
    numbered_probs = list(zip(range(1, 10), *zip(*probs)))
    final_out = {'map_html': out, 'probabilities': numbered_probs,
            'start_location': start_location}
    print('DEBUG, final_out', final_out)
    return final_out


def get_start_location(record, bundle):
    # Validate input
    # Make the start the first location.
    stationsdf = bundle['stations_bundle']['stationsdf']
    df = stationsdf[stationsdf['station_name'] 
                    == record['start station name']]
    if df.empty:
        return None
    else:
        return fu.subset(
            dict(df.iloc[0]), ['latlng', 'station_name'])


def make_csvdata(record, header):
    return ','.join([str(record[k]) for k in header])


def call_sagemaker(record):
    header = ['starttime',
            'start station name',
            'usertype',
            'birth year',
            'gender']

    if os.getenv('IN_LAMBDA'):
        try:
            return _call_inner(record, header)
        except botocore.exceptions.ClientError as e:
            error = {'error_detail': repr(e),
                     'error': e.__class__.__name__}
            
            hack_city_starttime = '2020-10-01 18:10:00'
            print('Grrr that weird bug, changing the starttime from ',
                    record['starttime'], ' to ', hack_city_starttime)
            print(error)
            record['starttime'] = hack_city_starttime
            return _call_inner(record, header)

    else:
        url = LOCAL_URL
        headers = {'Content-Type': 'text/csv'}
        # tripduration,starttime,stoptime,start station id,start station name,start station latitude,start station longitude,end station id,end station name,end station latitude,end station longitude,bikeid,usertype,birth year,gender\n

        csvdata = make_csvdata(record, header)
        print('DEBUG call_sagemaker, csvdata, ', csvdata)
        r = requests.post(url, data=csvdata, headers=headers)
        assert r.status_code/100 == 2, \
                'got this instead, r.status_code ' + str(r.status_code)

        return r.json()


def _call_inner(record, header):
    csvdata = make_csvdata(record, header)
    print('DEBUG call_sagemaker, csvdata, ', csvdata)

    endpoint = os.getenv('SAGEMAKER_ENDPOINT')
    client = boto3.client('sagemaker-runtime',
            region_name='us-east-1')
    #
    response = client.invoke_endpoint(
            EndpointName=endpoint,
            # Body=b'bytes'|file,
            Body=csvdata,
            ContentType='text/csv',
            Accept='string',
            # CustomAttributes='string'
            )
    what = response['Body'].read()
    return json.loads(what)


def map_probabilities(bundle, prob_vec, k=5):
    # so lambda downloads bundle from s3 live perhaps
    le = bundle['proc_bundle']['bundle']['proc_bundle']['le']
    classes = le.classes_.shape
    le.classes_ # 54

    top_k = sorted(list(zip(le.classes_, prob_vec)), key=lambda x: x[1], reverse=True)[:k] 
    return top_k


def blah_get_map(bundle, probs, start_location):
    stationsdf = bundle['stations_bundle']['stationsdf']
    df = pd.DataFrame(probs, columns=['neighborhood', 'prob'])

    # FIXME these are actually station latlng not center of neighborhoods
    #   maybe i can even highlight regions in google api? 
    locations = df.merge(
            stationsdf, on='neighborhood'
            ).drop_duplicates(subset='neighborhood')[['latlng', 'neighborhood']
                    ].to_dict(orient='records')

    out = fm.grab_final_thing([start_location] + locations)
    return out


def fetch_bundle():
    bundle = pickle.loads(fs3.read_s3_file(s3uri=BUNDLE_LOC_S3))
    return bundle
