

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

'''

import pandas as pd


from utils import distance_between_positions
import settings as s

def load_data(source_file=s.TRIPS_FILE):
    df = pd.read_csv(source_file)

    # df = df[:20]
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
            s.SPEED_COL_NAME: True, 
            s.AGE_COL_NAME: False,
            }

    if recalculate_dict[s.DISTANCE_TRAVELED_COL_NAME]:
        dist_travelled = calc_distance_travelled_col(df)
        df[s.DISTANCE_TRAVELED_COL_NAME] = pd.Series(dist_travelled)

    if recalculate_dict[s.SPEED_COL_NAME]:
        travel_speeds = calc_speeds(df)
        df[s.SPEED_COL_NAME] = pd.Series(travel_speeds)

    if recalculate_dict[s.AGE_COL_NAME]:
        df[s.AGE_COL_NAME] = 2015 - df[s.BIRTH_YEAR_COL]

    return df


if __name__ == '__main__':
    df = load_data('foo.csv')

    import ipdb; ipdb.set_trace()
    df = append_travel_stats(df)

    import ipdb; ipdb.set_trace()
    pass

    df.to_csv('foo.csv')

    plot_age_speed(df)

    plot_distance_trip_time(df)

