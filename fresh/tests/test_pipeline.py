


import requests
LOCAL_URL = 'http://127.0.0.1:8080/invocations'
LOCAL_PING_URL = 'http://127.0.0.1:8080/ping'

def test_docker_ping():
    url = LOCAL_PING_URL
    r = requests.get(url)
    assert r.status_code/100 == 2, \
            'got this instead, r.status_code ' + str(r.status_code)

def test_docker_duh():
    url = LOCAL_URL
    headers = {'Content-Type': 'text/csv'}
    # tripduration,starttime,stoptime,start station id,start station name,start station latitude,start station longitude,end station id,end station name,end station latitude,end station longitude,bikeid,usertype,birth year,gender\n

    header = ['starttime',
            'start station name',
            'usertype',
            'birth year',
            'gender']

    record = {
     'starttime': '2013-07-01 00:00:00',
     'start station name': 'E 47 St & 2 Ave',
     'usertype': 'Customer',
     'birth year': '1999',
     'gender': 0
     # 'start station id': 164,
     # 'start station latitude': 40.75323098,
     # 'start station longitude': -73.97032517,
     # 'bikeid': 16950,
     }

    csvdata = ','.join([str(record[k]) for k in header])

    r = requests.post(url, data=csvdata, headers=headers)
    assert r.status_code/100 == 2, \
            'got this instead, r.status_code ' + str(r.status_code)

