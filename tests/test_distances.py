


from utils import distance_between_positions
from decimal import Decimal

def test_distance_1():
    '''
from tests import test_distances
test_distances.test_distance_1()
    '''

    p1_latitude = Decimal(40.74972)
    p1_longitude = Decimal(-74.00295)

    p2_latitude = Decimal(40.74735)
    p2_longitude = Decimal(-73.99724)

    distance = distance_between_positions(
            p1_latitude, p1_longitude,
            p2_latitude, p2_longitude)

    print 'distance= ', distance 

