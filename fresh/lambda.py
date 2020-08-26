import os
import fresh.map as fm


import requests
LOCAL_URL = 'http://127.0.0.1:8080/invocations'

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

def map_probabilities():
    le = bundle['proc_bundle']['bundle']['proc_bundle']['le']
    print(le.classes_.shape)
    le.classes_ # 54

