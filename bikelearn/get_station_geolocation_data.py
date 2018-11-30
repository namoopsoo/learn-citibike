
import pandas as pd
import json
import time
import urllib2
import requests
import datetime
from copy import deepcopy

import settings as s

from data_store import connect_redis
redis_client = connect_redis()

POSTAL_CODE = 'postal_code'
SUBLOCALITY = 'sublocality'
NEIGHBORHOOD = 'neighborhood'
STATE_LABEL_CODE = 'administrative_area_level_1'
STATE = 'state' 
NY = 'NY'
STATION_NAME = 'station_name'
DF_STATIONS_COLUMNS = [STATION_NAME, POSTAL_CODE,
        SUBLOCALITY, NEIGHBORHOOD, STATE]

class NonSpecificAddress(Exception): pass

def validate_geo_result(geo_result_dict):
    return len(set([POSTAL_CODE,
        SUBLOCALITY, NEIGHBORHOOD, STATE])
        & set(geo_result_dict.keys())) == 4


def _parse_geocoding_result(raw_results_list):
    '''Take a raw googleapis.com geocoding query result and parse for locality data.
    
    Dont want the full complex result object. For now just want an easy dictionary
    to deal with afterwards. Also return the raw result for cacheability.
    
    NOTE: Assuming first result is the only one worth using.
    '''
    postal_code = None
    geo_results = {}
    for component in raw_results_list[0]['address_components']:
        if POSTAL_CODE in component['types']:
            geo_results[POSTAL_CODE] = component['long_name']
        if SUBLOCALITY in component['types']:
            geo_results[SUBLOCALITY] = component['long_name']
        if NEIGHBORHOOD in component['types']:
            geo_results[NEIGHBORHOOD] = component['long_name']
        if STATE_LABEL_CODE in component['types']:
            geo_results[STATE] = component['short_name']

    
    return {
        'raw_result': raw_results_list,
        'geo_results': geo_results }


def extract_lat_lng_from_response(geocode_response_results):
    '''Take a response, and return lat lang

    - Expecting it to have at least one result,
    - Using the first one if more

    '''
    first_result = geocode_response_results[0]
    coordinates_dict = first_result['geometry']['location']
    return coordinates_dict


def make_geo_url(address):
    # Use good encoding so intersection addresses with "&"
    #    don't add incorrect param separation into querystring
    address_encoded = urllib2.quote(address)
    assert '&' not in address_encoded
    url = 'https://maps.googleapis.com/maps/api/geocode/json?key={}&address={}'.format(
        s.GOOGLE_GEO_API_KEY, address_encoded)
    return url


def make_latlng_url(address):
    url = 'https://maps.googleapis.com/maps/api/geocode/json?key={}&latlng={}&result_type={}'.format(
        s.GOOGLE_GEO_API_KEY, address, 'neighborhood')
    return url


def make_url(address, request_type):
    if request_type == 'geo':
        return make_geo_url(address)

    if request_type == 'latlng':
        return make_latlng_url(address)

def get_geocoding_results(address, request_type, bypass_cache=False):
    assert request_type in ['geo', 'latlng']
    if not bypass_cache: 
        cached_response = redis_client.hget(s.GEO_RAW_RESULTS, address)
        if cached_response:
            print 'DEBUG, using cache for, ', address
            geo_result_dict = _parse_geocoding_result(
                    json.loads(cached_response)['results'])

            out = deepcopy(geo_result_dict)
            out.update(
                    {'cache_hit': True,
                        'levels': len(geo_result_dict),
                        'valid': validate_geo_result(geo_result_dict),
                        'address': address,})
            return out

    url = make_url(address, request_type)
    response = requests.get(url=url)
    assert '20' in str(response.status_code), 'hmm, {}, {}'.format(
            str(response.status_code), response.text)
    geocoding_result = response.json()

    #assert 'OK' == geocoding_result['status'], 'hmm, ' + str(geocoding_result)
    if not 'OK' == geocoding_result['status']:
        return {'valid': False, 'levels': 0,
                'google_response_status': geocoding_result['status']}

    annotated = _parse_geocoding_result(geocoding_result['results'])

    valid = validate_geo_result(annotated['geo_results'])
    if valid:
        redis_client.hset(s.GEO_RAW_RESULTS, address, response.text)

    annotated.update({'valid': valid, 'address': address,
        'levels': len(annotated['geo_results'])})
    return annotated


def _str_safe(name):
    try:
        str(name)
        return True
    except UnicodeEncodeError:
        return False


def get_address_geo_wrapper(address, bypass_cache=False):
    address_ny = address + ', NY'
    location_result = get_geocoding_results(address_ny, request_type='geo',
            bypass_cache=bypass_cache)
    if location_result.get('cache_hit') and not location_result.get('levels'):
        levels = len(location_result.get('geo_results'))
        if len(levels) == 4:
            return location_result

    if location_result['levels'] == 4:
        return location_result

    # XXX hmm... but only do this... if the aabove has postal code right?
    coordinates_dict = extract_lat_lng_from_response(
            location_result['raw_result'])
    address_latlng = '{lat},{lng}'.format(**coordinates_dict)
    another_location_result = get_geocoding_results(address_latlng,
            request_type='latlng',
            bypass_cache=bypass_cache)

    geo_result_dict = dict(
            location_result.get('geo_results').items())
            #+ another_location_result.get('geo_results').items()) 

    if another_location_result.get('geo_results'):
        geo_result_dict.update(
                another_location_result.get('geo_results'))
    out = {'geo_results': geo_result_dict,
            'levels': len(geo_result_dict),
            'valid': validate_geo_result(geo_result_dict),
            'address': address,
            'latlng': True
            }

    return out


def try_get_neighborhood(latlng):
    coordinates_dict = extract_lat_lng_from_response(location_result['raw_result'])

    address_latlng = '{lat},{lng}'.format(**coordinates_dict)
    another_location_result = get_geocoding_results(address_latlng, request_type='latlng')

    getgeo.extract_lat_lng_from_response(out.json()['results'])

    url += 'result_type=neighborhood'

    # => 


def get_station_geoloc_data(stations_list):
    '''
    Get geoloc data for stations in input json.

    Example:
    address = '2 Ave & E 58 St, NY'
    location_results = get_geocoding_results(address)
    '''
    results = []
    for address in stations_list:
        # Google api throttles when > 10 requests/sec
        time.sleep(0.11)

        parsed = get_address_geo_wrapper(address)
        out = {'station_name': address}
        out.update(parsed.get('geo_results', {}))
        out.update(parsed)
        results.append(out)

    df = pd.DataFrame.from_records(results)
    return df


def read_start_station_names(df):
    name = (s.START_STATION_NAME
            if s.START_STATION_NAME in df.columns.tolist()
            else s.START_STATION_NAME201110)
    return df[name].unique().tolist()


def extract_stations_from_files(filename=None, filenames=None):
    '''Given a citibike data filename, get list of stations
    '''
    if filename:
        filenames = [filename]
    return list(
            set(reduce(lambda x, y: x + y,
            [read_start_station_names(pd.read_csv(fn))
                for fn in filenames])))


def cleanse_cache_of_non_specific_data(dry_run=True):

    entries = redis_client.hkeys(s.GEO_RAW_RESULTS)
    whats_valid = []

    for address in entries:
        cached_response = json.loads(
                redis_client.hget(s.GEO_RAW_RESULTS, address))
        geo_result_list = cached_response.get('results', [])
        geo_result_dict = _parse_geocoding_result(geo_result_list)['geo_results']

        whats_valid.append({
            'valid': validate_geo_result(geo_result_dict),
            'address': address,
            'geo_result_dict': geo_result_dict,
            'levels': len(geo_result_dict)})
    if not dry_run:
        redis_client.hdel(s.GEO_RAW_RESULTS,
                *[x['address'] for x in whats_valid
                    if not x['valid']])
        
    return pd.DataFrame.from_records(whats_valid)



