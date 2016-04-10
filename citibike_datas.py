

'''
tripduration                              171
starttime                  10/1/2015 00:00:02
stoptime                   10/1/2015 00:02:54
start station id                          388
start station name           W 26 St & 10 Ave
start station latitude               40.74972
start station longitude             -74.00295
end station id                            494
end station name              W 26 St & 8 Ave
end station latitude                 40.74735
end station longitude               -73.99724
bikeid                                  24302
usertype                           Subscriber
birth year                               1973
gender                                      1
Name: 0, dtype: object

# 
Predictive power of time, age, gender and start location? 
    For the, 
        destination,
        trip duration,
        speed

... Maybe use more data ?


'''

import pandas as pd
from collections import OrderedDict 

from utils import distance_between_positions, get_start_time_bucket
from classify import (prepare_datas,
        build_classifier, run_predictions, run_metrics_on_predictions)
# from plottings import (plot_age_speed, plot_distance_trip_time)

from pipeline_data import append_travel_stats, load_data
import settings as s


def how_good_is_a_route(route):
    ''' Given the departures from the station at start of route,
    and arrivals to stations at end of route, based on their busy-ness,
    how good is a given route compared to others.
    '''
    pass

def get_total_number_destinations(df):
    ''' How many destination stations are there in the given dataset.

    '''

    stations_counts = df[s.END_STATION_NAME].value_counts()

    num_destinations = stations_counts.shape[0]

    return num_destinations

def build_classifier_to_predict_destination_station(df, definition):
    '''
from citibike_datas import (build_classifier_to_predict_destination_station)


definition = {
    'features': [
        s.START_STATION_ID,
        s.START_TIME_BUCKET,
        s.AGE_COL_NAME,
        s.GENDER,],
    'feature_encoding': {
        # 'borough': 1,
    }
    'label_col': s.END_STATION_ID,
}


df = load_data('data/201510-citibike-tripdata.csv.annotated.mini.02212016T1641.csv')

results = build_classifier_to_predict_destination_station(df, 
definition)

    '''

    # Extract only the relevant data columns,
    #     start time bucket (hour), age, gender and start location,
    pass

    datas = prepare_datas(df, features=definition['features'],
            feature_encoding=definition['feature_encoding'],
            label_col=definition['label_col'])

    # also need to account for filling in missing data in the holdout set.

    results = OrderedDict()

    classifier = build_classifier()
    classifier.fit(datas['X_train'], datas['y_train'])

    y_predictions = run_predictions(classifier, datas['X_train'])
    classif_metrics = run_metrics_on_predictions(datas['y_train'], y_predictions)
    results['training'] = classif_metrics

    y_predictions = run_predictions(classifier, datas['X_holdout'])
    classif_metrics = run_metrics_on_predictions(datas['y_holdout'], y_predictions)
    results['holdout'] = classif_metrics

    return results


def predict_destination(df):
    '''
    Given start time bucket (hour), age, gender and start location, 
        how many outputs are there, for the, 
            destination,
            trip duration,
            speed

    selecting the input conditions, use 'end station id', as the output,
        see which inputs are the most influential.

    (* Consider gridifying the start locations too based on neighborhoods.)

    NEXT STEPS:
    - create new dependent columns in the data, for, 
        (a) start time bucket , using utils.py:get_start_time_bucket()
    - Look at the value counts of the destinations for a given 
        (start time bucket, age, gender, start location),

        So for (start=Monday 0AM-1AM, 31year, F, 33rd&2Ave loc),

        how many destinations are there? And how does that compare to the total 
        number of possible destinations?
        
        - need a group by for that 4-tuple, 

    '''
    import ipdb; ipdb.set_trace()


    build_classifier_to_predict_destination_station(df)


def analyze_trip_destination_stats(df):

    # How many people are there in each 4-tuple bucket?
    #   => meaning, for the different source conditions, how many unique trips,
    #   were being made. But since this data is taken over a month period,
    #   , the very rare buckets may mean very rare starting conditions in general,
    #   for which there is not much predictive opportunity. 
    #       
    source_grpby = df.groupby([
        s.START_STATION_NAME, 
        s.START_TIME_BUCKET, 
        s.AGE_COL_NAME,
        s.GENDER, 
        ], 
        #, as_index=False
        )

    # count the (start time bucket, age, gender, start location) ..
    group_sizes = source_grpby.sizes()
    group_sizes.to_csv('unique_starting_conditions.012416T1804.csv')
    

    how_many_groups = source_grpby.size().shape
    # df.groupby(['start station name', 'starttime_bucket', 'age', 'gender']).size().shape
    # (703276,)


    df.groupby([
        s.START_STATION_NAME, 
        s.END_STATION_NAME, ]).size().shape[0]
    # .... 96382  many combinations...


    # And what about when taking destinations into account as well ?
    #   - Can we identify commuters? What would be a good visualization for seeing 
    #   if the same people are making these trips?
    #
    #
    trips_grpby = df.groupby([
        s.START_STATION_NAME, 
        s.END_STATION_NAME, 
        s.START_TIME_BUCKET, 
        s.AGE_COL_NAME,
        s.GENDER, 
        ], )
    how_many_groups = trips_grpby.size().shape
    # ...  
    #   => Given that for 10/2015, there are 1065766 trips,
    #   and 955606 distinct kinds of trips, for the people making these trips.



if __name__ == '__main__':
    df = load_data('foo.csv', num_rows=2000)

    import ipdb; ipdb.set_trace()
    df = append_travel_stats(df)

    import ipdb; ipdb.set_trace()
    pass


    predict_destination(df)

    df.to_csv('foo.csv')

    plot_age_speed(df)

    plot_distance_trip_time(df)

