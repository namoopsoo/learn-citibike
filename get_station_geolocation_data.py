
# coding: utf-8

# In[89]:

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


# In[96]:

def get_station_geoloc_data(stations_json_filename):
    '''
    Get geoloc data for stations in json

    '''
    stations = json.load(open(stations_json_filename))
    
    df = pd.DataFrame({'station_name': stations}, columns=df_stations_columns)
    
    return df

    num_indices = 
    
    for address in df['station_name']:
        time.sleep(0.5)
        
        # Attach NY.
        address += ', NY'
        location_results = get_geoloc_data(address)
        for location_result in location_results:
            
            for key in location_results:
                df.iloc[i][key] =
        
        
        
        
    
    return df


# In[8]:

stations_json_filename = 'data/start_stations_103115.json'


# In[97]:

stations_df = get_station_geoloc_data(stations_json_filename)


# In[98]:

stations_df.head()


# In[102]:

stations_df.iloc[1]['city'] = 'rook'


# In[103]:

stations_df.iloc[1]


# In[23]:

len(stations)


# In[29]:

stations.columns


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


# In[83]:

get_geoloc_data('1 Ave & E 15 St, NY')


# In[30]:

stations.head()


# In[54]:

df2 = pd.DataFrame({'a':[5,6,7,8], 'b':[4,3,6,7]})


# In[80]:

df2


# In[79]:

df2.loc[5] = [7,0]


# In[ ]:



