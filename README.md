## Citibike Project: Can your Destination be Predicted

### Summary
The full report is [here](project%20report.ipynb), but in summary, this project was a progression of trying to determine how input data on a trip can be used to help understood where people go. At first, using data from a single month inlate 2015, of `804125` many trips, using a simple classifier, with something like `~476` citibike stations, the prediction accuracy was really bad, at roughly `3%`. The realization came that having `476` classes, intuitively, would be a hard probelm to start with, so the next step taken was to pull in the Google Geolocation data on the stations, to try to get earlier wins. 

Switching to sublocalities, a.k.a. boroughs, the prediction accuracy was in the 90s. The accuracy for a neighborhood (_of which there are some 28_) was in the 40s, and so that was the general target chosen for the majority of this project. A few different approaches were taken, detailed in this report. Different sizes of training data were used for comparison. Also different levels of time bucketing preprocessing,  were used as well. Specifically, one hot encoding the starting neighborhood increased the accuracy from around `0.40` to closer to `0.47`. 

The final approach tested during the experimentation, was actually not a change in preprocessing, a change in the learning algorithm, nor in the selection of the data, but in the evaluation metric. Given the classification problem of predicting one of `28` neighborhoods, instead of evaluating accuracy, a so-called _Rank k Accuracy_ was also explored, to see how looking at the top `k=1`, `k=2`, `k=3`, `k=4` and `k=5` predictions could evaluate the performance. The result showed clearly better than incremental improvement, with the rank `k=3` accuracy being around `0.76`, for example. This of course is _cheating!_, but it was interesting to explore this, because in the real world, we do not always require the first choice to be correct. In the example of a search engine, we tend to expect a good answer to our query to be within the top four or five of the results displayed.

### Thoughts for future improvement
Given the opportunity to flesh out the problem more, I think using more data would be near the top of the list of approaches to try. Experimenting with different sizes of training data did not show any interesting affect on the performance of the classifier, but there are now several more months of data available to play around witih.  In discussing with a few colleagues, seasonality would also be a really good feature to consider. Time bucketing was explored to a limited extent, but the day of the week nor the month of the year was not explored. There may also be many other datasets which can be joined with this one to bolster the information available, including information about the weather or perhaps other demographic attributes available. A more thorough comparison of algorithms should also be considered.

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

stations_json_filename = 'datas/start_stations_103115.json'

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
import classify
import annotate_geolocation as annotate_geo
import settings as s

stations_df_filename = os.path.join(s.DATAS_DIR, 'start_stations_103115.fuller.csv')
stations_df = pd.read_csv(stations_df_filename, index_col=0, dtype={'postal_code': str})


octoberdf = pd.read_csv(s.DATAS_DIR + '/201510-citibike-tripdata.csv')

next_df = annotate_geo.annotate_df_with_geoloc(octoberdf, stations_df, noisy_nonmatches=False)

and_age_df = pl.annotate_age(next_df)
more_df = pl.annotate_time_features(and_age_df)

simpledf, label_encoders = annotate_geo.make_medium_simple_df(more_df)
simpledf.to_csv(s.DATAS_DIR + '/201510-citibike-tripdata.medium-simple.csv', index_label='index')


# simple train/test split
train_df, holdout_df = classify.simple_split(simpledf)
train_df.to_csv(s.DATAS_DIR + '/201510-citibike-tripdata.medium-simple-train.csv', index_label='index')
holdout_df.to_csv(s.DATAS_DIR + '/201510-citibike-tripdata.medium-simple-holdout.csv', index_label='index')


# reading later..
simpledf = pd.read_csv(s.DATAS_DIR + '/201510-citibike-tripdata.medium-simple.csv', index_col='index')

```

