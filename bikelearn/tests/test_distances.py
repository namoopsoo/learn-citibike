import unittest
import bikelearn.utils as bu
from decimal import Decimal

class TestDistance(unittest.TestCase):
    def test_distance_1(self):
        '''
    from tests import test_distances
    test_distances.test_distance_1()
        '''

        p1_latitude = 40.74972
        p1_longitude = -74.00295

        p2_latitude = 40.74735
        p2_longitude = -73.99724

        distance = bu.distance_between_positions(
                p1_latitude, p1_longitude,
                p2_latitude, p2_longitude)

        print 'distance= ', distance 

class TestBoxArea(unittest.TestCase):

    def test_1(self):

        geo = {u'bounds': {u'northeast': {u'lat': 40.77410690000001,
            u'lng': -73.95889869999999},
            u'southwest': {u'lat': 40.72686849999999, u'lng': -74.0089488}},
            u'location': {u'lat': 40.7549309, u'lng': -73.9840195},
            u'location_type': u'APPROXIMATE',
            u'viewport': {u'northeast': {u'lat': 40.77410690000001,
                u'lng': -73.95889869999999},
                u'southwest': {u'lat': 40.72686849999999, u'lng': -74.0089488}}}


        area1 = bu.latlng_box_area(geo['bounds'])
        area2 = bu.latlng_box_area(geo['viewport'])
