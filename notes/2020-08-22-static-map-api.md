

```python
import pandas as pd
import os
from importlib import reload

import fresh.map as fm
import fresh.predict_utils as fpu
```




    <module 'fresh.predict_utils' from '/opt/program/fresh/predict_utils.py'>




```python


loc = '/opt/program/artifacts/2020-08-19T144654Z/all_bundle.joblib'
bundle = fpu.load_bundle(loc)

print(bundle.keys())

# ok for instance, 
bundle['neighborhoods_bundle']


```

    dict_keys(['notebook', 'model_bundle', 'proc_bundle', 'neighborhoods_bundle'])





    {'original_filename': '/opt/program/datas/stations/stations-2018-12-04-c.csv',
     'neighborhoods': ['Alphabet City',
      'Battery Park City',
      'Bedford-Stuyvesant',
      'Bloomingdale',
      'Boerum Hill',
      'Bowery',
      'Broadway Triangle',
      'Brooklyn Heights',
      'Brooklyn Navy Yard',
      'Carnegie Hill',
      'Carroll Gardens',
      'Central Park',
      'Chelsea',
      'Chinatown',
      'Civic Center',
      'Clinton Hill',
      'Cobble Hill',
      'Columbia Street Waterfront District',
      'Downtown Brooklyn',
      'Dumbo',
      'East Harlem',
      'East Village',
      'East Williamsburg',
      'Financial District',
      'Flatiron District',
      'Fort Greene',
      'Fulton Ferry District',
      'Garment District',
      'Governors Island',
      'Gowanus',
      'Gramercy Park',
      'Greenpoint',
      'Greenwich Village',
      "Hell's Kitchen",
      'Hudson Square',
      'Hunters Point',
      'Kips Bay',
      'Korea Town',
      'Lenox Hill',
      'Lincoln Square',
      'Little Italy',
      'Long Island City',
      'Lower East Side',
      'Lower Manhattan',
      'Meatpacking District',
      'Midtown',
      'Midtown East',
      'Midtown West',
      'Murray Hill',
      'NoHo',
      'NoMad',
      'Nolita',
      'Park Slope',
      'Peter Cooper Village',
      'Prospect Heights',
      'Prospect Park',
      'Red Hook',
      'Rose Hill',
      'SoHo',
      'Stuyvesant Heights',
      'Stuyvesant Town',
      'Sunset Park',
      'Sutton Place',
      'Theater District',
      'Tribeca',
      'Tudor City',
      'Two Bridges',
      'Ukrainian Village',
      'Union Square',
      'Upper East Side',
      'Upper West Side',
      'Vinegar Hill',
      'West Village',
      'Williamsburg',
      'Yorkville']}




```python
import fresh.map as fm
```


```python
import pandas as pd
localdir = '/opt/program'
stationsdf = pd.read_csv(f'{localdir}/datas/stations/stations-2018-12-04-c.csv',
                        index_col=0)
```


```python
locations = [{'latlng': x['latlng']} 
             for x in stationsdf.iloc[:3]]
locations
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-10-656e882d9853> in <module>
    ----> 1 locations = [{'latlng': x['latlng']} for x in stationsdf.iloc[:3]]
          2 locations


    <ipython-input-10-656e882d9853> in <listcomp>(.0)
    ----> 1 locations = [{'latlng': x['latlng']} for x in stationsdf.iloc[:3]]
          2 locations


    TypeError: string indices must be integers



```python
list(stationsdf.iloc[:3].to_records())
```




    [(0, 'Bank St & Hudson St', 40.73652889, -74.00618026, '40.73652889,-74.00618026', 'West Village', 10014, 'NY', 'Manhattan'),
     (1, 'E 59 St & Sutton Pl', 40.75849116, -73.95920622, '40.75849116,-73.95920622', 'Sutton Place', 10022, 'NY', 'Manhattan'),
     (2, 'E 37 St & Lexington Ave', 40.748238, -73.978311, '40.748238,-73.978311', 'Murray Hill', 10016, 'NY', 'Manhattan')]




```python
os.path.exists('tempmap.html')

```




    True




```python
reload(fm)
locations = stationsdf.iloc[:3].to_dict(orient='records') # list
locations = [{'latlng': '40.6924182926,-73.989494741'}]
url = fm.make_url_from_locations(locations)
secret = os.environ['GOOGLE_CLIENT_SECRET']
url = fm.sign_url(url, secret=secret)

url = fm.escape_url(url)
img = fm.make_img_tag(url)

with open(f'tempmap_{random.randint(int(1e4), int(1e5))}.html', 
          'w') as fd:
    fd.write(img)
```


```python
import random
random.randint(int(1e4), int(1e5))
```




    68926




```python
# os.environ['GOOGLE_GEO_API_KEY'] = 
# os.environ['GOOGLE_CLIENT_SECRET'] = 
```


```python
secret = os.environ['GOOGLE_CLIENT_SECRET']
fm.base64.urlsafe_b64decode(secret)
```




    b'$\x88vI_\xfb\xbd\xfd\\\xd7?\xb5\xc4(\xe3\x9f\x15\xb7\x16\xf4'




```python
stationsdf
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>station_name</th>
      <th>start station latitude</th>
      <th>start station longitude</th>
      <th>latlng</th>
      <th>neighborhood</th>
      <th>postal_code</th>
      <th>state</th>
      <th>sublocality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bank St &amp; Hudson St</td>
      <td>40.736529</td>
      <td>-74.006180</td>
      <td>40.73652889,-74.00618026</td>
      <td>West Village</td>
      <td>10014</td>
      <td>NY</td>
      <td>Manhattan</td>
    </tr>
    <tr>
      <th>1</th>
      <td>E 59 St &amp; Sutton Pl</td>
      <td>40.758491</td>
      <td>-73.959206</td>
      <td>40.75849116,-73.95920622</td>
      <td>Sutton Place</td>
      <td>10022</td>
      <td>NY</td>
      <td>Manhattan</td>
    </tr>
    <tr>
      <th>2</th>
      <td>E 37 St &amp; Lexington Ave</td>
      <td>40.748238</td>
      <td>-73.978311</td>
      <td>40.748238,-73.978311</td>
      <td>Murray Hill</td>
      <td>10016</td>
      <td>NY</td>
      <td>Manhattan</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9 Ave &amp; W 14 St</td>
      <td>40.740583</td>
      <td>-74.005509</td>
      <td>40.7405826,-74.00550867</td>
      <td>Meatpacking District</td>
      <td>10014</td>
      <td>NY</td>
      <td>Manhattan</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Broadway &amp; W 53 St</td>
      <td>40.763441</td>
      <td>-73.982681</td>
      <td>40.76344058,-73.98268129</td>
      <td>Midtown West</td>
      <td>10019</td>
      <td>NY</td>
      <td>Manhattan</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>676</th>
      <td>Berkeley Pl &amp; 7 Ave</td>
      <td>40.675147</td>
      <td>-73.975232</td>
      <td>40.6751468387,-73.9752320945</td>
      <td>Park Slope</td>
      <td>11217</td>
      <td>NY</td>
      <td>Brooklyn</td>
    </tr>
    <tr>
      <th>677</th>
      <td>Fulton St &amp; Adams St</td>
      <td>40.692418</td>
      <td>-73.989495</td>
      <td>40.6924182926,-73.989494741</td>
      <td>Downtown Brooklyn</td>
      <td>11201</td>
      <td>NY</td>
      <td>Brooklyn</td>
    </tr>
    <tr>
      <th>678</th>
      <td>E 76 St &amp; 3 Ave</td>
      <td>40.772249</td>
      <td>-73.958421</td>
      <td>40.7722485377,-73.9584213495</td>
      <td>Lenox Hill</td>
      <td>10075</td>
      <td>NY</td>
      <td>Manhattan</td>
    </tr>
    <tr>
      <th>679</th>
      <td>Bressler</td>
      <td>40.646538</td>
      <td>-74.016588</td>
      <td>40.6465383671,-74.0165877342</td>
      <td>Sunset Park</td>
      <td>11232</td>
      <td>NY</td>
      <td>Brooklyn</td>
    </tr>
    <tr>
      <th>681</th>
      <td>4 Ave &amp; 2 St</td>
      <td>40.674613</td>
      <td>-73.985011</td>
      <td>40.6746134225,-73.9850114286</td>
      <td>Park Slope</td>
      <td>11215</td>
      <td>NY</td>
      <td>Brooklyn</td>
    </tr>
  </tbody>
</table>
<p>681 rows × 8 columns</p>
</div>


