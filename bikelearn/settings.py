
GEO_RAW_RESULTS = 'geolocation_raw_results'


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
START_DAY = 'start_day'
START_HOUR = 'start_hour'


NEW_START_POSTAL_CODE = 'start_postal_code'
NEW_START_STATE = 'start_state'
NEW_START_BOROUGH = 'start_sublocality'
NEW_START_NEIGHBORHOOD = 'start_neighborhood'
NEW_START_STATION_NAME = 'start_station_name'

NEW_START_STATION_COLS = [ NEW_START_POSTAL_CODE,
        NEW_START_BOROUGH,
        NEW_START_NEIGHBORHOOD, NEW_START_STATION_NAME] 

NEW_END_POSTAL_CODE = 'end_postal_code'
NEW_END_STATE = 'end_state'
NEW_END_BOROUGH = 'end_sublocality'
NEW_END_NEIGHBORHOOD = 'end_neighborhood'
NEW_END_STATION_NAME = 'end_station_name'

NEW_END_STATION_COLS = [NEW_END_POSTAL_CODE,
        NEW_END_STATE, NEW_END_BOROUGH,
        NEW_END_NEIGHBORHOOD, NEW_END_STATION_NAME]

ALL_END_STATION_COLS = NEW_END_STATION_COLS + [
        END_STATION_LATITUDE_COL, END_STATION_LONGITUDE_COL,
        END_STATION_NAME, END_STATION_ID]


ALL_COLUMNS = ['tripduration', 'starttime', 'stoptime', 'start station id', 'start station name', 'start station latitude', 'start station longitude', 'end station id', 'end station name', 'end station latitude', 'end station longitude', 'bikeid', 'usertype', 'birth year', 'gender']



DATAS_DIR = 'datas'
TRIPS_FILE = DATAS_DIR + '/' + '201510-citibike-tripdata.csv'

PLOTS_DIR = 'plots'

REDIS_SERVER = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 3
REDIS_TIMEOUT = 3

CLASSIFICATION_RESULTS_KEY = 'classification_results_key'


from secret_settings import *

