
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
df_stations_columns = [STATION_NAME, POSTAL_CODE,SUBLOCALITY, NEIGHBORHOOD, STATE]


geo_raw_responses_dir = 'data/geo_raw_responses/'


def _parse_geocoding_result(geocoding_result):
    '''Take a raw googleapis.com geocoding query result and parse for locality data.
    
    Dont want the full complex result object. For now just want an easy dictionary
    to deal with afterwards. Also return the raw result for cacheability.'''

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

        if geo_results.get(STATE) != NY:
            # false match. crap.
            return None
                
    else:
        return None
    
    return {
        'raw_result': raw_results,
        'geo_results': geo_results
        }


def get_geocoding_results(address, overwrite_cache=False):
    '''Get to geocoding results directly from googleapis.com
    
    Use a redis cache to store responses, so we don't have to query for the 
    same data when performing multiple runs of this function. But in some cases,
    we will want to overwrite the result if we suspect there are changes in what we
    are storing or how we are storing the result.
    
    Also, initially, we were fetching the incorrect data because the url being used
    included unencoded '&' ampersand within the address, which was corrupting the query.
    And in that case, the cache results needed to be completely overwritten.
    '''
   
    # look up on redis first
    GEO_RAW_RESULTS = 'geolocation_raw_results'
    cached_response = redis_client.hget(GEO_RAW_RESULTS, address)

    if cached_response and not overwrite_cache:
        print 'using cache for ', address
        geocoding_result = json.loads(cached_response)
    else:
        try:
            address_encoded = address.replace(' ', '+')
            # Use good encoding so intersection addresses with "&"
            #    don't add incorrect param separation into querystring
            address_encoded = urllib2.quote(address)
            url = 'https://maps.googleapis.com/maps/api/geocode/json?key={}&address={}'.format(
                s.GOOGLE_GEO_API_KEY,
                address_encoded)
    
            response = requests.get(url=url)
            if not '20' in str(response.status_code):
                return None
            else:
                geocoding_result = json.loads(response.text)
                redis_client.hset(GEO_RAW_RESULTS, address, response.text)
        except:
            return None
    
    results = _parse_geocoding_result(geocoding_result)
    return results
        

address = 'Myrtle Ave & Marcy Ave, NY'
results2 = get_geocoding_results(address)




# In[9]:

results2['geo_results']


# In[10]:

def get_station_geoloc_data(stations_json_filename):
    '''
    Get geoloc data for stations in input json.

Example:
address = '2 Ave & E 58 St, NY'
location_results = get_geocoding_results(address)

    '''
    stations = json.load(open(stations_json_filename))
    
    df = pd.DataFrame({'station_name': stations}, columns=df_stations_columns)
    
    for i in df.index:
        address = df.iloc[i]['station_name']
        # Google api throttles when > 10 requests/sec
        time.sleep(0.11)
        
        # Attach NY. Cheating here, but this increases the chances we
        #   only deal with NY here to limit false matches.
        address_ny = address + ', NY'
        location_result = get_geocoding_results(address_ny)
        if not location_results:
            continue

        for key, val in location_result['geo_results'].items():
            df.iloc[i][key] = val
    
    return df

