

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


START_STATION_LATITUDE_COL = 'start station latitude'
START_STATION_LONGITUDE_COL = 'start station longitude'
END_STATION_LATITUDE_COL = 'end station latitude'
END_STATION_LONGITUDE_COL = 'end station longitude'

DATAS_DIR = 'data'
TRIPS_FILE = DATAS_DIR + '/' + '201510-citibike-tripdata.csv'

from utils import distance_between_positions

def load_data():
    df = pd.read_csv(TRIPS_FILE)

    smaller_df = df[:20]

    return smaller_df 

def calc_distance_travelled_col(df):


    distance = distance_between_positions(
            df[START_STATION_LATITUDE_COL],
            df[START_STATION_LONGITUDE_COL],
            df[END_STATION_LATITUDE_COL],
            df[END_STATION_LONGITUDE_COL],
            )

def append_travel_stats(df):

    distance_traveled_col_name = 'distance travelled'
    speed_col_name = 'speed'

    dist_travelled = calc_distance_travelled_col(df)

    df[distance_traveled_col_name] = pd.Series()


    return df 

if __name__ == '__main__':
    import ipdb; ipdb.set_trace()
    df = load_data()

    df = append_travel_stats(df)

