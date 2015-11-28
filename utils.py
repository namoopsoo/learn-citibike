
from math import cos, pi, fabs, sqrt
from decimal import Decimal

EARTH_RADIUS = 3958.8
EARTH_CIRCUMFERENCE = EARTH_RADIUS * 2 * pi
MILES_PER_DEGREE_LAT = EARTH_CIRCUMFERENCE / 360

PRECISION = Decimal('0.001')



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


