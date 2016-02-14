

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
 

from utils import distance_between_positions, get_start_time_bucket
from classify import (prepare_datas,
        build_classifier, run_predictions)
# from plottings import (plot_age_speed, plot_distance_trip_time)
import settings as s

def load_data(source_file=s.TRIPS_FILE, num_rows=None):
    ''' Load subscriber data.
    '''
    df = pd.read_csv(source_file)

    if num_rows:
        df = df[:num_rows]

    df = df[df[s.USER_TYPE_COL] == s.USER_TYPE_SUBSCRIBER]

    return df

def calc_distance_travelled_col(df):
    '''

    TODO... apply in dataframe notation.
            df[START_STATION_LATITUDE_COL],
            df[START_STATION_LONGITUDE_COL],
            df[END_STATION_LATITUDE_COL],
            df[END_STATION_LONGITUDE_COL],

    '''

    values = df.as_matrix(columns=[
        s.START_STATION_LATITUDE_COL,
        s.START_STATION_LONGITUDE_COL,
        s.END_STATION_LATITUDE_COL,
        s.END_STATION_LONGITUDE_COL])

    distances = []

    for row in values:
        distance = distance_between_positions(*row)

        distances.append(distance)

    return distances

def calc_speeds(df):
    '''
    miles/hour = X miles/seconds * 60sec/min * 60min/hour    

    '''
    values = df.as_matrix(columns=[
        s.DISTANCE_TRAVELED_COL_NAME,
        s.TRIP_DURATION_COL])

    speeds = []

    for row in values:
        speed = 60*60*row[0]/row[1]

        speeds.append(speed)
    
    return speeds

def append_travel_stats(df):
    
    recalculate_dict = {
            s.DISTANCE_TRAVELED_COL_NAME: False,
            s.SPEED_COL_NAME: False, 
            s.AGE_COL_NAME: False,
            s.START_TIME_BUCKET: True,
            }

    if recalculate_dict[s.DISTANCE_TRAVELED_COL_NAME]:
        dist_travelled = calc_distance_travelled_col(df)
        df[s.DISTANCE_TRAVELED_COL_NAME] = pd.Series(dist_travelled)

    if recalculate_dict[s.SPEED_COL_NAME]:
        travel_speeds = calc_speeds(df)
        df[s.SPEED_COL_NAME] = pd.Series(travel_speeds)

    if recalculate_dict[s.AGE_COL_NAME]:
        df[s.AGE_COL_NAME] = 2015 - df[s.BIRTH_YEAR_COL]

    import ipdb; ipdb.set_trace()

    if recalculate_dict[s.START_TIME_BUCKET]:
        time_buckets = calculate_start_time_buckets(df)
        df[s.START_TIME_BUCKET] = pd.Series(time_buckets)

    return df

def calculate_start_time_buckets(df):

    values = df.as_matrix(columns=[
        s.START_TIME])

    start_time_buckets = []

    for row in values:
        buck = get_start_time_bucket(row[0])
        start_time_buckets.append(buck)

    
    return start_time_buckets


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

def build_classifier_to_predict_destination_station(df):
    '''

    '''

    # Extract only the relevant data columns,
    #     start time bucket (hour), age, gender and start location, 
    pass

    datas = prepare_datas(df, features=[
        s.START_STATION_ID, 
        s.START_TIME_BUCKET, 
        s.AGE_COL_NAME,
        s.GENDER,], label_col=s.END_STATION_ID)

    # also need to account for filling in missing data in the holdout set.

    classifier = build_classifier()


    classifier.fit(datas['X_train'], datas['y_train'])

    run_predictions(classifier, datas['X_train'], datas['y_train'])
    run_predictions(classifier, datas['X_holdout'], datas['y_holdout'])


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

