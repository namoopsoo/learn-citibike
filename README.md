

### Annotate citibike data with Google Geocoding tags

#### Quick prerequisites
* set your API key
```python
# secret_settings.py
GOOGLE_GEO_API_KEY = 'blah'
```

#### Take the base for a quick spin
```python
import get_station_geolocation_data as getgeo

address = "W 26 St & 10 Ave"

data = getgeo.get_geocoding_results(address)

```

