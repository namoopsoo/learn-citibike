
import pandas as pd
import json
import time
import urllib2
import requests
import datetime


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
DF_STATIONS_COLUMNS = [STATION_NAME, POSTAL_CODE,SUBLOCALITY, NEIGHBORHOOD, STATE]


def _parse_geocoding_result(geocoding_result):
    '''Take a raw googleapis.com geocoding query result and parse for locality data.
    
    Dont want the full complex result object. For now just want an easy dictionary
    to deal with afterwards. Also return the raw result for cacheability.
    
    NOTE: Assuming first result is the only one worth using.
    '''

    raw_results = geocoding_result['results']
    
    if raw_results:
        
        postal_code = None
        geo_results = {}
        for component in raw_results[0]['address_components']:
            if POSTAL_CODE in component['types']:
                geo_results[POSTAL_CODE] = component['long_name']
            if SUBLOCALITY in component['types']:
                geo_results[SUBLOCALITY] = component['long_name']
            if NEIGHBORHOOD in component['types']:
                geo_results[NEIGHBORHOOD] = component['long_name']
            if STATE_LABEL_CODE in component['types']:
                geo_results[STATE] = component['short_name']

                
    else:
        return None
    
    return {
        'raw_result': raw_results,
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
    url = 'https://maps.googleapis.com/maps/api/geocode/json?key={}&address={}'.format(
        s.GOOGLE_GEO_API_KEY, address)
    return url


def make_latlng_url(address):
    url = 'https://maps.googleapis.com/maps/api/geocode/json?key={}&latlng={}'.format(
        s.GOOGLE_GEO_API_KEY, address)
    return url


def get_geocoding_results(address, request_type, overwrite_cache=False):
    '''Get to geocoding results directly from googleapis.com
    
    Use a redis cache to store responses, so we don't have to query for the 
    same data when performing multiple runs of this function. But in some cases,
    we will want to overwrite the result if we suspect there are changes in what we
    are storing or how we are storing the result.
    
    Also, initially, we were fetching the incorrect data because the url being used
    included unencoded '&' ampersand within the address, which was corrupting the query.
    And in that case, the cache results needed to be completely overwritten.
    '''
    assert request_type in ['geo', 'latlng']
   
    # look up on redis first
    cached_response = redis_client.hget(s.GEO_RAW_RESULTS, address)

    if cached_response and not overwrite_cache:
        print 'using cache for ', address
        geocoding_result = json.loads(cached_response)
    else:
        try:
            address_encoded = address.replace(' ', '+')
            # Use good encoding so intersection addresses with "&"
            #    don't add incorrect param separation into querystring
            address_encoded = urllib2.quote(address)
            if request_type == 'geo':
                url = make_geo_url(address)
            elif request_type == 'latlng':
                url = make_latlng_url(address)
            response = requests.get(url=url)

            if not '20' in str(response.status_code):
                return None
            else:
                geocoding_result = json.loads(response.text)
                redis_client.hset(s.GEO_RAW_RESULTS, address, response.text)
        except Exception as e:
            print 'this address crapped out ', address
            print e
            print 'continuing...\n'
            return None
    
    results = _parse_geocoding_result(geocoding_result)
    return results


def _str_safe(name):
    try:
        str(name)
        return True
    except UnicodeEncodeError:
        return False


def get_address_geo_wrapper(address):
    '''
    First try standard query

    but if it is vague, grab geo coords from it and retry it.
    '''
    # Attach NY. Cheating here, but this increases the chances we
    #   only deal with NY here to limit false matches.
    address_ny = address + ', NY'
    location_result = get_geocoding_results(address_ny, request_type='geo')

    if location_result is None:
        print 'station %s couldnt be processed' % address_ny
        return

    geo_results = location_result['geo_results']

    if  geo_results.get(NEIGHBORHOOD) is not None:
        # NEIGHBORHOOD, POSTAL_CODE
        if geo_results.get(STATE) != NY:
            # false match. crap.
            print 'station %s couldnt be processed, getting non-NY response.' % address_ny
            return
        else:
            return geo_results
    else:
        # Try getting a result from lat long instead
        coordinates_dict = extract_lat_lng_from_response(location_result['raw_result'])

        address_latlng = '{lat},{lng}'.format(**coordinates_dict)
        another_location_result = get_geocoding_results(address_latlng, request_type='latlng')
        if another_location_result is None:
            print 'station %s couldnt be processed, even after trying lat/lng %s' % (
                    address_ny, address_latlng)
            return
        else:
            #
            another_geo_result = another_location_result['geo_results']
            if another_geo_result.get(STATE) != NY:
                # false match. crap.
                print 'station %s couldnt be processed, getting non-NY response.(lat/lng %s)' % (
                        address_ny, address_latlng)
                return
            else:
                return another_geo_result


def get_station_geoloc_data(stations_json_filename):
    '''
    Get geoloc data for stations in input json.

    Example:
    address = '2 Ave & E 58 St, NY'
    location_results = get_geocoding_results(address)

    '''
    stations = json.load(open(stations_json_filename))

    stations_cleaned = [name for name in stations
            if _str_safe(name)]
    
    df = pd.DataFrame({'station_name': stations_cleaned}, columns=DF_STATIONS_COLUMNS)
    
    for i in df.index:
        address = df.iloc[i]['station_name']
        # Google api throttles when > 10 requests/sec
        time.sleep(0.11)

        location_result = get_address_geo_wrapper(address)
        if location_result is not None:
            for key, val in location_result.items():
                df.iloc[i][key] = val
    
    return df

def extract_stations_from_data(filename):
    '''Given a citibike data filename, get list of stations
    '''
    # df = pd.read_csv(filename)

    # stations_json_filename

    # df = pd.read_csv(s.DATAS_DIR + '/201510-citibike-tripdata.csv')
    df = pd.read_csv(s.DATAS_DIR + '/' + filename)

    stations_list = df[s.START_STATION_NAME].unique().tolist()

    return stations_list




