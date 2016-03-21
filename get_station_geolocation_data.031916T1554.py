
# coding: utf-8

# In[134]:

from geolocation.main import GoogleMaps
import pandas as pd
import json
import time
import urllib2
import requests



# In[42]:

# The Google Maps Geocoding API
# https://developers.google.com/maps/documentation/geocoding/intro#GeocodingResponses


# In[ ]:

import settings as s

from data_store import connect_redis
redis_client = connect_redis()

POSTAL_CODE = 'postal_code'
SUBLOCALITY = 'sublocality'
NEIGHBORHOOD = 'neighborhood'
STATE = 'administrative_area_level_1'
NY = 'NY'
STATION_NAME = 'station_name'
df_stations_columns = [STATION_NAME, POSTAL_CODE,SUBLOCALITY, NEIGHBORHOOD, STATE]


# In[ ]:

geo_raw_responses_dir = 'data/geo_raw_responses/'


# In[8]:

google_maps = GoogleMaps(api_key=s.GOOGLE_GEO_API_KEY)


# In[ ]:

def _parse_geocoding_result(geocoding_result):    

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
            if STATE in component['types']:
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


# In[ ]:

def get_geocoding_results(address, overwrite_cache=False):
    '''Get to geocoding results directly
    '''
   
    # look up on redis first
    GEO_RAW_RESULTS = 'geolocation_raw_results'
    cached_response = redis_client.hget(GEO_RAW_RESULTS, address)

    if cached_response and not overwrite_cache:
        geocoding_result = json.loads(cached_response)
    else:
        print 'Not using cache'
        address_encoded = address.replace(' ', '+')
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
    
    results = _parse_geocoding_result(geocoding_result)
    return results
        


# In[ ]:

address = 'Myrtle Ave & Marcy Ave, NY'
results2 = get_geocoding_results(address, overwrite_cache=True)




# In[149]:

def get_station_geoloc_data(stations_json_filename):
    '''
    Get geoloc data for stations in json

    '''
    import pdb; pdb.set_trace()
    stations = json.load(open(stations_json_filename))
    
    df = pd.DataFrame({'station_name': stations}, columns=df_stations_columns)
    
    stations_multiple_matches = {}
    
    for i in df.index:
        address = df.iloc[i]['station_name']
        time.sleep(0.11)
        
        # Attach NY.
        address_ny = address + ', NY'
        location_result = get_geocoding_results(address_ny)
        if not location_results:
            continue

        for key, val in location_result['geo_results'].items():
            df.iloc[i][key] = val
    
    return df


# In[41]:

address = 'Myrtle Ave , Marcy Ave, Brooklyn, NY'
address = '753 Myrtle Ave, Brooklyn, NY'
address = '5-25 47 Rd, Queens, NY'
# address = '2 Ave & E 58 St, NY'
location_results = get_geoloc_data(address)

location_results


# In[12]:

stations_json_filename = 'data/start_stations_103115.json'


# In[13]:

stations = json.load(open(stations_json_filename))


# In[150]:

len(stations
   )


# In[ ]:

stations_df = get_station_geoloc_data(stations_json_filename)


# In[115]:

stations_df.head()


# In[121]:

stations_df.to_excel('data/stations_geoloc_030516.xls')


# In[105]:

for i in stations_df.head().index:
    address = stations_df.iloc[i]['station_name']
    print i, address


# In[1]:

stations_df.iloc[96]


# In[23]:

len(stations)


# In[29]:

stations.columns


# In[30]:

stations.head()


# In[54]:

df2 = pd.DataFrame({'a':[5,6,7,8], 'b':[4,3,6,7]})


# In[79]:

df2.loc[5] = [7,0]


# In[ ]:



