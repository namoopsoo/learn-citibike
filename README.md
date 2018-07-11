## Citibike Project: Can your Destination be Predicted

### Summary
* Report: https://github.com/namoopsoo/learn-citibike/blob/master/project%20report.ipynb
* Report: [report](project%20report.ipynb)

### Example data
```
"tripduration","starttime","stoptime","start station id","start station name","start station latitude","start station longitude","end station id","end station name","end station latitude","end station longitude","bikeid","usertype","birth year","gender"
"171","10/1/2015 00:00:02","10/1/2015 00:02:54","388","W 26 St & 10 Ave","40.749717753","-74.002950346","494","W 26 St & 8 Ave","40.74734825","-73.99723551","24302","Subscriber","1973","1"
"593","10/1/2015 00:00:02","10/1/2015 00:09:55","518","E 39 St & 2 Ave","40.74780373","-73.9734419","438","St Marks Pl & 1 Ave","40.72779126","-73.98564945","19904","Subscriber","1990","1"
"233","10/1/2015 00:00:11","10/1/2015 00:04:05","447","8 Ave & W 52 St","40.76370739","-73.9851615","447","8 Ave & W 52 St","40.76370739","-73.9851615","17797","Subscriber","1984","1"
"250","10/1/2015 00:00:15","10/1/2015 00:04:25","336","Sullivan St & Washington Sq","40.73047747","-73.99906065","223","W 13 St & 7 Ave","40.73781509","-73.99994661","23966","Subscriber","1984","1"
"528","10/1/2015 00:00:17","10/1/2015 00:09:05","3107","Bedford Ave & Nassau Ave","40.72311651","-73.95212324","539","Metropolitan Ave & Bedford Ave","40.71534825","-73.96024116","16246","Customer","","0"
"440","10/1/2015 00:00:17","10/1/2015 00:07:37","3107","Bedford Ave & Nassau Ave","40.72311651","-73.95212324","539","Metropolitan Ave & Bedford Ave","40.71534825","-73.96024116","23698","Customer","","0"
```

## Notes

### Annotate citibike data with Google Geocoding tags

#### Quick prerequisites
* set your API key
```python
# secret_settings.py
GOOGLE_GEO_API_KEY = 'blah'
```

#### Take the base for a quick spin
```python
import get_station_geolocation_data as getgeo

address = "W 26 St & 10 Ave"

data = getgeo.get_geocoding_results(address)



```
* get citibike station geo data
```python
import get_station_geolocation_data as getgeo

stations_list = getgeo.extract_stations_from_data('201510-citibike-tripdata.csv')

stations_json_filename = 'data/start_stations_103115.json'

with open(stations_json_filename, 'w') as fd:
	json.dump(stations_list, fd, indent=4)

# now use that base stations file to grab all the results from google api...
stations_df = getgeo.get_station_geoloc_data(stations_json_filename)

stations_df.to_csv(os.path.join(s.DATAS_DIR, 'start_stations_103115.csv'))
```
* make a simple output annotated
```python

octoberdf = pd.read_csv(s.DATAS_DIR + '/201510-citibike-tripdata.csv')

# make geo-annotated df..
import annotate_geolocation as annotate_geo
next_df = annotate_geo.annotate_df_with_geoloc(octoberdf, stations_df, noisy_nonmatches=False)

simpledf = annotate_geo.make_dead_simple_df(next_df)

simpledf.to_csv(s.DATAS_DIR + '/201510-citibike-tripdata.simple.csv')


```
* medium simple output annotated df,
```python
import os
import pandas as pd
 import pipeline_data as pl
import annotate_geolocation as annotate_geo
import settings as s

stations_df_filename = os.path.join(s.DATAS_DIR, 'start_stations_103115.fuller.csv')
stations_df = pd.read_csv(stations_df_filename, index_col=0, dtype={'postal_code': str})


octoberdf = pd.read_csv(s.DATAS_DIR + '/201510-citibike-tripdata.csv')

next_df = annotate_geo.annotate_df_with_geoloc(octoberdf, stations_df, noisy_nonmatches=False)

and_age_df = pl.annotate_age(next_df)
more_df = pl.annotate_time_features(and_age_df)

simpledf = annotate_geo.make_medium_simple_df(more_df)
simpledf.to_csv(s.DATAS_DIR + '/201510-citibike-tripdata.medium-simple.csv', index_label='index')


# simple train/test split
train_df, holdout_df = classify.simple_split(simpledf)
train_df.to_csv(s.DATAS_DIR + '/201510-citibike-tripdata.medium-simple-train.csv', index_label='index')
holdout_df.to_csv(s.DATAS_DIR + '/201510-citibike-tripdata.medium-simple-holdout.csv', index_label='index')


# reading later..
simpledf = pd.read_csv(s.DATAS_DIR + '/201510-citibike-tripdata.medium-simple.csv', index_col='index')

```

