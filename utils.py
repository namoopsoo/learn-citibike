
from math import cos, pi
from decimal import Decimal

EARTH_RADIUS = Decimal('3958.8')

def distance_between_positions(
        point_1_lat,
        point_1_long,
        point_2_lat,
        point_2_long,):

    '''
Approximately, at 40 lat, each longit degree is about 52miles/deg.
    And approx, at 41 lat, the rate is about 51miles/deg.

Also, the cross-sectional radius at degree x, Rcosx

w/ earth radius, R = 3958.8 mi
'''

def deg_to_rad(deg):
    rad = pi * deg / 180
    return rad


def miles_per_long_degree_at_lat_degree(degree_latitude):

    degree_latitude_rads = deg_to_rad(degree_latitude)

    radius_at_degree = EARTH_RADIUS*Decimal(cos(degree_latitude_rads))

    circumference = 2*Decimal(pi)*radius_at_degree

    miles_per_long_degree = circumference/Decimal(360)

    return miles_per_long_degree.quantize(Decimal('0.001'))


