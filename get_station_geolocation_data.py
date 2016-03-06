
# coding: utf-8

# In[108]:

from geolocation.main import GoogleMaps
import pandas as pd
import json
import time


# In[3]:

import settings as s


# In[4]:

google_maps = GoogleMaps(api_key=s.GOOGLE_GEO_API_KEY)


# In[86]:

df_stations_columns = ['station_name', 'city', 'lat', 'lng', 
                     'postal_code', 'route', 'street_number', 'formatted_address']


# In[81]:

def get_geoloc_data(address):
    
    location_info = google_maps.search(location=address)
    
    results = []
    location_results = location_info.all()

    for result in location_results:
        location_dict = {}
        for attr in ['city', 'lat', 'lng', 
                     'postal_code', 'route', 'street_number', 'formatted_address']:
            val = getattr(result, attr)
            location_dict[attr] = val
        results.append(location_dict)
        
    return results


# In[117]:

def get_station_geoloc_data(stations_json_filename):
    '''
    Get geoloc data for stations in json

    '''
    stations = json.load(open(stations_json_filename))
    
    df = pd.DataFrame({'station_name': stations}, columns=df_stations_columns)
    
    stations_multiple_matches = {}
    
    for i in df.index:
        address = df.iloc[i]['station_name']
        time.sleep(0.11)
        
        # Attach NY.
        address_ny = address + ', NY'
        location_results = get_geoloc_data(address_ny)
        if len(location_results) == 1:
            location_result = location_results[0]
        elif len(location_results) > 1:
            stations_multiple_matches[address] = location_results
            location_result = location_results[0]
        else:
            continue

        for key, val in location_result.items():
            df.iloc[i][key] = val
        print i, address, df.iloc[i]['city']
    
    return df, stations_multiple_matches


# In[8]:

stations_json_filename = 'data/start_stations_103115.json'


# In[118]:

stations_df, stations_multiple_matches = get_station_geoloc_data(stations_json_filename)


# In[115]:

stations_df.head()


# In[121]:

stations_df.to_excel('data/stations_geoloc_030516.xls')


# In[105]:

for i in stations_df.head().index:
    address = stations_df.iloc[i]['station_name']
    print i, address


# In[103]:

stations_df.iloc[1]


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



