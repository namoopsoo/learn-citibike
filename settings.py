

from secret_settings import *

START_STATION_LATITUDE_COL = 'start station latitude'
START_STATION_LONGITUDE_COL = 'start station longitude'
START_STATION_NAME = 'start station name'
START_STATION_ID = 'start station id'
END_STATION_LATITUDE_COL = 'end station latitude'
END_STATION_LONGITUDE_COL = 'end station longitude'
END_STATION_NAME = 'end station name'
END_STATION_ID = 'end station id'
TRIP_DURATION_COL = "tripduration"
BIRTH_YEAR_COL = "birth year"
GENDER = 'gender'
USER_TYPE_COL = "usertype"
USER_TYPE_SUBSCRIBER = "Subscriber"
START_TIME = "starttime"

# New Columns
DISTANCE_TRAVELED_COL_NAME = 'distance travelled'
SPEED_COL_NAME = 'speed'
AGE_COL_NAME = 'age'
START_TIME_BUCKET = "starttime_bucket"


DATAS_DIR = 'data'
TRIPS_FILE = DATAS_DIR + '/' + '201510-citibike-tripdata.csv'

PLOTS_DIR = 'plots'

REDIS_SERVER = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 3
REDIS_TIMEOUT = 3

