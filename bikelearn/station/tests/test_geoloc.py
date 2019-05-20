import unittest
import json

import bikelearn.get_station_geolocation_data as getgeo

class Foo(unittest.TestCase):
    def test_invalid(self):
        fn = 'data/raw.clinton-flushing.json'
        fn = 'bikelearn/station/tests/data/raw.clinton-flushing.json'
        with open(fn) as fd: data = json.load(fd)

        parsed = getgeo._parse_geocoding_result(data['results'])
        assert not getgeo.validate_geo_result(parsed['geo_results'])


class MoreComplexGeolocIntegrationTest(unittest.TestCase):
    def test_2ave_e105st(self):
        address = '2 Ave & E 105 St'



        parsed = getgeo.get_address_geo_wrapper(address, bypass_cache=True)

        assert parsed['levels'] == 4

        assert parsed['geo_results']['neighborhood'] == 'East Harlem'

