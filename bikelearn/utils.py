
from math import cos, pi, fabs, sqrt
from decimal import Decimal
import numpy as np

from copy import deepcopy
import datetime

import settings as s
from os import path

EARTH_RADIUS = 3958.8
EARTH_CIRCUMFERENCE = EARTH_RADIUS * 2 * pi
MILES_PER_DEGREE_LAT = EARTH_CIRCUMFERENCE / 360

PRECISION = Decimal('0.001')

STARTTIME_REGEX = "%m/%d/%Y %H:%M:%S"
STARTTIME_REGEX_201610 = '%Y-%m-%d %H:%M:%S'


def _quantize(x):

    return x.quantize(PRECISION)

def distance_between_positions(
        point_1_lat,
        point_1_long,
        point_2_lat,
        point_2_long,):
    '''
Approximately, at 40 lat, each longit degree is about 52miles/deg.
    And approx, at 41 lat, the rate is about 51miles/deg.

Also, the cross-sectional radius at degree x, Rcosx

'''

#    point_1_lat = Decimal(point_1_lat)
#    point_1_long = Decimal(point_1_long)
#    point_2_lat = Decimal(point_2_lat)
#    point_2_long = Decimal(point_2_long)

    d_betw_longitude_degrees = distance_betw_longitude_degrees(
        point_1_lat,
        point_1_long,
        point_2_long)

    d_betw_latitude_degrees = distance_betw_lat_degrees(
            point_1_lat, 
            point_2_lat)

    d_diagonal = sqrt(
            d_betw_latitude_degrees**2 +
            d_betw_longitude_degrees**2)

    return d_diagonal

def distance_betw_lat_degrees(lat_1_degree, lat_2_degree):
    ''' Orthogonal distance between two consecutive latitude degrees,
    is eath circumference divided by 360.
    '''
    difference = fabs(lat_1_degree - lat_2_degree)

    distance_in_miles = MILES_PER_DEGREE_LAT * difference 
    return distance_in_miles

def distance_betw_longitude_degrees(
        latitude,
        longitude_1_degree,
        longitude_2_degree):
    ''' Approximate distance at short distances.  '''
    miles_per_degree = miles_per_long_degree_at_lat_degree(latitude)

    difference = fabs(longitude_1_degree - longitude_2_degree)

    distance_in_miles = miles_per_degree * difference 

    return distance_in_miles


def deg_to_rad(deg):
    rad = pi * deg / 180
    return rad


def miles_per_long_degree_at_lat_degree(degree_latitude):

    degree_latitude_rads = deg_to_rad(degree_latitude)

    radius_at_degree = EARTH_RADIUS*cos(degree_latitude_rads)

    circumference = 2*pi*radius_at_degree

    miles_per_long_degree = circumference/360

    return miles_per_long_degree


def get_start_time_bucket(start_time):
    '''
For 24x7 = 168 hours in a week, starting from Monday midnight,
    assign buckets,  0 through 167

    Monday, 00:00:00 to 00:59:59, => 0
    Monday, 01:00:00 to 01:59:59, => 1
    '''

    # What day of week?
    try:
        d = datetime.datetime.strptime(str(start_time), STARTTIME_REGEX)
    except ValueError:
        d = datetime.datetime.strptime(str(start_time), STARTTIME_REGEX_201610)

    weekday = d.weekday()  # Monday is 0
    
    hour = d.hour

    # Delta from the closest Monday...
    delta = 24*weekday + hour

    return delta


def get_start_day(start_time):
    try:
        d = datetime.datetime.strptime(str(start_time), STARTTIME_REGEX)
    except ValueError:
        d = datetime.datetime.strptime(str(start_time), STARTTIME_REGEX_201610)

    weekday = d.weekday()  # Monday is 0
    return weekday    


def get_start_hour(start_time):
    try:
        d = datetime.datetime.strptime(str(start_time), STARTTIME_REGEX)
    except ValueError:
        d = datetime.datetime.strptime(str(start_time), STARTTIME_REGEX_201610)

    hour = d.hour
    return hour


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


def which_col_have_nulls(df):
    ''' Return list of cols in df with have at least one null.
    '''
    have_nulls = []

    for col in df.columns:
        if df[col].isnull().any():
            how_many_null = df[col][df[col].isnull()].shape[0]
            num_rows = df.shape[0]
            assert how_many_null <= num_rows

            percent_null = int(Decimal(100*how_many_null)/Decimal(num_rows))

            have_nulls.append((col, percent_null))

    return have_nulls


def dump_np_array(X, basename):

    filename = datetime.datetime.now().strftime(
            path.join(s.DATAS_DIR, basename + '.%m%d%YT%H%M.csv'))

    np.savetxt(filename, X, delimiter=",")


def latlng_box_area(box):
    if box is None: return None

    {u'northeast': {u'lat': 40.77410690000001,
            u'lng': -73.95889869999999},
            u'southwest': {u'lat': 40.72686849999999, u'lng': -74.0089488}}

    point_1_lat = box['northeast']['lat']
    point_1_long = box['northeast']['lng']
    point_2_lat = box['southwest']['lat']
    point_2_long = box['southwest']['lng']

    d_betw_longitude_degrees = distance_betw_longitude_degrees(
        point_1_lat,
        point_1_long,
        point_2_long)

    d_betw_latitude_degrees = distance_betw_lat_degrees(
            point_1_lat, 
            point_2_lat)

    return d_betw_longitude_degrees * d_betw_latitude_degrees 


def print_bundle(bundle):
    acopy = deepcopy(bundle)

    m = acopy['evaluation']['test_metrics']['confusion_matrix']
    acopy['evaluation']['test_metrics']['confusion_matrix'] = len(m)

    m = acopy['evaluation']['validation_metrics']['confusion_matrix']
    acopy['evaluation']['validation_metrics']['confusion_matrix'] = len(m)

    # [printable(acopy, key) for key in acopy.keys()]
    return acopy

