

```python
import pandas as pd
```


```python
repodir = '/Users/michal/LeDropbox/Dropbox/Code/repo'
tripsdf = pd.read_csv(f'{repodir}/data/citibike/2013-07 - Citi Bike trip data.csv')
stationsdf = pd.read_csv(f'{repodir}/learn-citibike/datas/stations/stations-2018-12-04-c.csv',
                        index_col=0)
```


```python
tripsdf.head()

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
      <th>tripduration</th>
      <th>starttime</th>
      <th>stoptime</th>
      <th>start station id</th>
      <th>start station name</th>
      <th>start station latitude</th>
      <th>start station longitude</th>
      <th>end station id</th>
      <th>end station name</th>
      <th>end station latitude</th>
      <th>end station longitude</th>
      <th>bikeid</th>
      <th>usertype</th>
      <th>birth year</th>
      <th>gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>634</td>
      <td>2013-07-01 00:00:00</td>
      <td>2013-07-01 00:10:34</td>
      <td>164</td>
      <td>E 47 St &amp; 2 Ave</td>
      <td>40.753231</td>
      <td>-73.970325</td>
      <td>504</td>
      <td>1 Ave &amp; E 15 St</td>
      <td>40.732219</td>
      <td>-73.981656</td>
      <td>16950</td>
      <td>Customer</td>
      <td>\N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1547</td>
      <td>2013-07-01 00:00:02</td>
      <td>2013-07-01 00:25:49</td>
      <td>388</td>
      <td>W 26 St &amp; 10 Ave</td>
      <td>40.749718</td>
      <td>-74.002950</td>
      <td>459</td>
      <td>W 20 St &amp; 11 Ave</td>
      <td>40.746745</td>
      <td>-74.007756</td>
      <td>19816</td>
      <td>Customer</td>
      <td>\N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>178</td>
      <td>2013-07-01 00:01:04</td>
      <td>2013-07-01 00:04:02</td>
      <td>293</td>
      <td>Lafayette St &amp; E 8 St</td>
      <td>40.730287</td>
      <td>-73.990765</td>
      <td>237</td>
      <td>E 11 St &amp; 2 Ave</td>
      <td>40.730473</td>
      <td>-73.986724</td>
      <td>14548</td>
      <td>Subscriber</td>
      <td>1980</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1580</td>
      <td>2013-07-01 00:01:06</td>
      <td>2013-07-01 00:27:26</td>
      <td>531</td>
      <td>Forsyth St &amp; Broome St</td>
      <td>40.718939</td>
      <td>-73.992663</td>
      <td>499</td>
      <td>Broadway &amp; W 60 St</td>
      <td>40.769155</td>
      <td>-73.981918</td>
      <td>16063</td>
      <td>Customer</td>
      <td>\N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>757</td>
      <td>2013-07-01 00:01:10</td>
      <td>2013-07-01 00:13:47</td>
      <td>382</td>
      <td>University Pl &amp; E 14 St</td>
      <td>40.734927</td>
      <td>-73.992005</td>
      <td>410</td>
      <td>Suffolk St &amp; Stanton St</td>
      <td>40.720664</td>
      <td>-73.985180</td>
      <td>19213</td>
      <td>Subscriber</td>
      <td>1986</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
stationsdf.head()
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
  </tbody>
</table>
</div>




```python
tripsdf.head().iloc[0]
```




    tripduration                               634
    starttime                  2013-07-01 00:00:00
    stoptime                   2013-07-01 00:10:34
    start station id                           164
    start station name             E 47 St & 2 Ave
    start station latitude                 40.7532
    start station longitude               -73.9703
    end station id                             504
    end station name               1 Ave & E 15 St
    end station latitude                   40.7322
    end station longitude                 -73.9817
    bikeid                                   16950
    usertype                              Customer
    birth year                                  \N
    gender                                       0
    Name: 0, dtype: object




```python
mdf = tripsdf[['start station name', 'end station name']
            ].merge(stationsdf[['station_name', 'neighborhood']], 
                    left_on='start station name',
                    right_on='station_name'
                   ).rename(columns={'neighborhood': 'start_neighborhood'}
                           ).merge(stationsdf[['station_name', 'neighborhood']],
                                  left_on='end station name',
                                   right_on='station_name'
                                  ).rename(columns={'neighborhood': 'end_neighborhood'})
```


```python
tripsdf.shape, mdf.shape
```




    ((843416, 15), (843416, 6))




```python
mdf.iloc[0]
```




    start station name    E 47 St & 2 Ave
    end station name      1 Ave & E 15 St
    station_name_x        E 47 St & 2 Ave
    start_neighborhood       Midtown East
    station_name_y        1 Ave & E 15 St
    end_neighborhood      Stuyvesant Town
    Name: 0, dtype: object




```python
statsdf = mdf[['start_neighborhood', 
               'end_neighborhood']].groupby(by=['start_neighborhood', 
               'end_neighborhood']).size().reset_index().rename(columns={0: 'count'})
```


```python
statsdf['count'].sum()
```




    843416




```python
# What is the most popular destination overall?
mdf[['end_neighborhood']].groupby(by='end_neighborhood').size().reset_index(
            ).rename(columns={0: 'count'}).sort_values(by='count', ascending=False)
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
      <th>end_neighborhood</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>Chelsea</td>
      <td>92721</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Midtown East</td>
      <td>41296</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Greenwich Village</td>
      <td>40157</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Tribeca</td>
      <td>39614</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Hell's Kitchen</td>
      <td>37192</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Financial District</td>
      <td>36062</td>
    </tr>
    <tr>
      <th>52</th>
      <td>West Village</td>
      <td>35993</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Midtown West</td>
      <td>30617</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Lower East Side</td>
      <td>26753</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Midtown</td>
      <td>26216</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Kips Bay</td>
      <td>24865</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Battery Park City</td>
      <td>24230</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Theater District</td>
      <td>21648</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Ukrainian Village</td>
      <td>21165</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Alphabet City</td>
      <td>18885</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Murray Hill</td>
      <td>18124</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Fort Greene</td>
      <td>16678</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Flatiron District</td>
      <td>16067</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Garment District</td>
      <td>15557</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Lower Manhattan</td>
      <td>15120</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bowery</td>
      <td>14947</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Gramercy Park</td>
      <td>13935</td>
    </tr>
    <tr>
      <th>15</th>
      <td>East Village</td>
      <td>13860</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Downtown Brooklyn</td>
      <td>13525</td>
    </tr>
    <tr>
      <th>42</th>
      <td>SoHo</td>
      <td>13417</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Chinatown</td>
      <td>12613</td>
    </tr>
    <tr>
      <th>53</th>
      <td>Williamsburg</td>
      <td>12197</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Stuyvesant Town</td>
      <td>11610</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Union Square</td>
      <td>11456</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Nolita</td>
      <td>11394</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Civic Center</td>
      <td>11178</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Hudson Square</td>
      <td>9644</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Lincoln Square</td>
      <td>9320</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Brooklyn Heights</td>
      <td>8630</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Meatpacking District</td>
      <td>7881</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Clinton Hill</td>
      <td>7494</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Rose Hill</td>
      <td>6366</td>
    </tr>
    <tr>
      <th>37</th>
      <td>NoMad</td>
      <td>6060</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Central Park</td>
      <td>5811</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Dumbo</td>
      <td>5606</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Two Bridges</td>
      <td>5120</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bedford-Stuyvesant</td>
      <td>4510</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Korea Town</td>
      <td>3576</td>
    </tr>
    <tr>
      <th>36</th>
      <td>NoHo</td>
      <td>3546</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Boerum Hill</td>
      <td>3372</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Peter Cooper Village</td>
      <td>2892</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Little Italy</td>
      <td>2874</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Brooklyn Navy Yard</td>
      <td>2575</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Fulton Ferry District</td>
      <td>2157</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Park Slope</td>
      <td>2040</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Columbia Street Waterfront District</td>
      <td>1904</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Tudor City</td>
      <td>1222</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Sutton Place</td>
      <td>891</td>
    </tr>
    <tr>
      <th>51</th>
      <td>Vinegar Hill</td>
      <td>833</td>
    </tr>
  </tbody>
</table>
</div>




```python
mdf['start_neighborhood'].unique().shape, mdf['end_neighborhood'].unique().shape
```




    ((54,), (54,))




```python
from scipy.special import softmax
import numpy as np
```


```python
x = np.array([[1, 0.5, 0.2, 3],
               [1,  -1,   7, 3],
               [2,  12,  13, 3]])
softmax(x, axis=1)[0].sum()
print('x[0]', x[0])
softmax(x[0])
```

    x[0] [1.  0.5 0.2 3. ]





    array([0.10587707, 0.06421769, 0.04757363, 0.78233161])




```python

def make_probs(xdf):
    xdf['prob'] = softmax(xdf['count'])
    xdf['prob2'] = xdf['prob'].map(lambda x:round(x, ndigits=4))
    #return xdf[['end_neighborhood', 'prob']].T
    #blahdf = statsdf[statsdf['start_neighborhood'] == 'Alphabet City'].copy()
    #blahdf['prob'] = softmax(blahdf['count'])
    #statsdf.groupby(by='start_neighborhood').apply()
    return pd.DataFrame(dict(
        list(xdf[['end_neighborhood', 'prob2']].to_records(index=False))),
                       index=[0])



```


```python
import datetime; import pytz
def utc_ts():
    return datetime.datetime.utcnow(
        ).replace(tzinfo=pytz.UTC).strftime('%Y-%m-%dT%H%M%SZ')
utc_ts()
```




    '2020-06-06T034021Z'




```python
hmmdf = statsdf.groupby(by='start_neighborhood').apply(make_probs)
statsdf.to_csv(f'/Users/michal/Downloads/2020-05-31--citilearn/{utc_ts()}-statsdf.csv')
hmmdf.to_csv(f'/Users/michal/Downloads/2020-05-31--citilearn/{utc_ts()}-hmmdf.csv')


```


```python
hmmdf
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
      <th></th>
      <th>Alphabet City</th>
      <th>Battery Park City</th>
      <th>Bedford-Stuyvesant</th>
      <th>Boerum Hill</th>
      <th>Bowery</th>
      <th>Brooklyn Heights</th>
      <th>Brooklyn Navy Yard</th>
      <th>Central Park</th>
      <th>Chelsea</th>
      <th>Chinatown</th>
      <th>...</th>
      <th>Sutton Place</th>
      <th>Theater District</th>
      <th>Tribeca</th>
      <th>Tudor City</th>
      <th>Two Bridges</th>
      <th>Ukrainian Village</th>
      <th>Union Square</th>
      <th>Vinegar Hill</th>
      <th>West Village</th>
      <th>Williamsburg</th>
    </tr>
    <tr>
      <th>start_neighborhood</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Alphabet City</th>
      <th>0</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Battery Park City</th>
      <th>0</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Bedford-Stuyvesant</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Boerum Hill</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Bowery</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Brooklyn Heights</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Brooklyn Navy Yard</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Central Park</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Chelsea</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Chinatown</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Civic Center</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.982</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Clinton Hill</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Columbia Street Waterfront District</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Downtown Brooklyn</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Dumbo</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.9933</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>East Village</th>
      <th>0</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Financial District</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Flatiron District</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Fort Greene</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Fulton Ferry District</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Garment District</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Gramercy Park</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Greenwich Village</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Hell's Kitchen</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Hudson Square</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Kips Bay</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Korea Town</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Lincoln Square</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0001</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Little Italy</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Lower East Side</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Lower Manhattan</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Meatpacking District</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Midtown</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Midtown East</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Midtown West</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Murray Hill</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>NoHo</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>NoMad</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Nolita</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Park Slope</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Peter Cooper Village</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Rose Hill</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>1.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>SoHo</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Stuyvesant Town</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Sutton Place</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Theater District</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Tribeca</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Tudor City</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0025</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Two Bridges</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Ukrainian Village</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Union Square</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Vinegar Hill</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>West Village</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Williamsburg</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>54 rows × 54 columns</p>
</div>



#### So a super basic model that just predicts using the top prior probabity
- Let's use that to choose some metrics and try to evaluate a model
- So with this 54 neighborhood setup, we can say, a prediction on an input, 
is a vector of the probability of the 54 output neighborhoods. And we can make the evaluation
metric the multilabel logloss.



```python

from sklearn.model_selection import train_test_split # (*arrays, **options)
```


```python
X, y 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```


```python
from sklearn.metrics import log_loss
import numpy as np
log_loss(["spam", "ham", "ham", "spam"],
         [[.1, .9], [.9, .1], [.8, .2], [.35, .65]])

```




    0.21616187468057912




```python
# Expecting near 0 for a perfect prediction
perfect_pred = np.array([[0, 0, 1],
                   [1, 0, 0],
                   [1, 0, 0],
                   [0, 0, 1],
                   [0, 1, 0]]) # [ham, jam, spam]
log_loss(["spam", "ham", "ham", "spam", "jam"],
         perfect_pred)

```




    2.1094237467877998e-15




```python
# ok cool and using numberic labels okay too?
perfect_pred = np.array([[0, 0, 1],
                   [1, 0, 0],
                   [1, 0, 0],
                   [0, 0, 1],
                   [0, 1, 0]]) # [ham, jam, spam]
log_loss([3, 1, 1, 3, 2],
         perfect_pred)

```




    2.1094237467877998e-15




```python
# and some noisy predictions? 
noisy_pred = np.array([[.05, .05, .9],
                   [.95, 0.05, 0],
                   [.9, 0.1, 0],
                   [0.05, .05, .9],
                   [0, 1, 0]]) # [ham, jam, spam]
log_loss([3, 1, 1, 3, 2],
         noisy_pred)

```




    0.07347496827220674




```python
# and some noisy predictions? 
noisy_pred = np.array([[.05, .05, .9, 0],
                   [.95, 0.05, 0, 0],
                   [.9, 0.1, 0, 0],
                   [0.05, .05, .9, 0],
                   [0, 1, 0, 0]]) # [ham, jam, spam]
log_loss([3, 1, 1, 3, 2],
         noisy_pred,
        labels=[1,2,3,4])

```




    0.07347496827220787



#### Train this super dumb baseline model


```python
def train(X, y):
    


```


```python
np.array(["hi"])
```




    array(['hi'], dtype='<U2')



#### Manual cross validation
- Doing this manually just because I had already written the train/fit using pandas
- And I want to just quickly do this


```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

X, y = datasets.load_iris(return_X_y=True)
X.shape, y.shape

```




    ((150, 4), (150,))




```python

from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, X, y, cv=5)
scores
```




    array([0.96666667, 1.        , 0.96666667, 0.96666667, 1.        ])




```python
pdb.runcall(cross_val_score, clf, X, y, cv=5)

```

    > /usr/local/miniconda3/envs/pandars3/lib/python3.7/site-packages/sklearn/model_selection/_validation.py(394)cross_val_score()
    -> scorer = check_scoring(estimator, scoring=scoring)
    (Pdb) l
    389  	    :func:`sklearn.metrics.make_scorer`:
    390  	        Make a scorer from a performance metric or loss function.
    391  	
    392  	    """
    393  	    # To ensure multimetric format is not supported
    394  ->	    scorer = check_scoring(estimator, scoring=scoring)
    395  	
    396  	    cv_results = cross_validate(estimator=estimator, X=X, y=y, groups=groups,
    397  	                                scoring={'score': scorer}, cv=cv,
    398  	                                return_train_score=False,
    399  	                                n_jobs=n_jobs, verbose=verbose,
    (Pdb) q



```python
class FooFlassifiler():
    def __init__(self):
        return
    def fit(self, X, y):
        pass
    def score(self, X, y):
        return .9
    def get_params(self, deep):
        return {}
    
    
```


```python
clf = FooFlassifiler()
scores = cross_val_score(clf, X, y, cv=5)

```


```python
scores
```




    array([0.9, 0.9, 0.9, 0.9, 0.9])




```python
X, y = mdf[['start_neighborhood']].values, mdf['end_neighborhood'].tolist()
```


```python
pd.DataFrame(X[:5], columns=['start_neighborhood']), y[:5]
```




    (  start_neighborhood
     0       Midtown East
     1       Midtown East
     2       Midtown East
     3       Midtown East
     4       Midtown East,
     ['Stuyvesant Town',
      'Stuyvesant Town',
      'Stuyvesant Town',
      'Stuyvesant Town',
      'Stuyvesant Town'])




```python
np.vstack((X[:5, 0], y[:5])).T
#X[:5,0]
```




    array([['Midtown East', 'Stuyvesant Town'],
           ['Midtown East', 'Stuyvesant Town'],
           ['Midtown East', 'Stuyvesant Town'],
           ['Midtown East', 'Stuyvesant Town'],
           ['Midtown East', 'Stuyvesant Town']], dtype=object)




```python
np.array(y[:5]).T
```




    array(['Stuyvesant Town', 'Stuyvesant Town', 'Stuyvesant Town',
           'Stuyvesant Town', 'Stuyvesant Town'], dtype='<U15')




```python
pd.DataFrame(np.vstack((X[:5, 0], y[:5])).T, columns=['start_neighborhood',
                                                   'end_neighborhood'])
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
      <th>start_neighborhood</th>
      <th>end_neighborhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Midtown East</td>
      <td>Stuyvesant Town</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Midtown East</td>
      <td>Stuyvesant Town</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Midtown East</td>
      <td>Stuyvesant Town</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Midtown East</td>
      <td>Stuyvesant Town</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Midtown East</td>
      <td>Stuyvesant Town</td>
    </tr>
  </tbody>
</table>
</div>




```python
split_data = model.iloc[:5].to_dict(orient='split')
lookup = {a: np.array(split_data['data'])[i] for (i, a) in enumerate([x[0] for x in split_data['index']])}
```


```python
len(split_data['columns'])
```




    54




```python
def dumb_fit(X, y):
    mdf = X_y_to_mydf(X, y)
    statsdf = mdf[['start_neighborhood', 
               'end_neighborhood']].groupby(by=['start_neighborhood', 
               'end_neighborhood']).size().reset_index().rename(columns={0: 'count'})
    hmmdf = statsdf.groupby(by='start_neighborhood').apply(make_probs).fillna(0)
    
    split_data = hmmdf.to_dict(orient='split')
    lookup_dict = {a: np.array(split_data['data'])[i] 
              for (i, a) in enumerate([x[0] for x in split_data['index']])}
    labels = split_data['columns']
    return hmmdf, lookup_dict, labels
    
def X_y_to_mydf(X, y):
    return pd.DataFrame(np.vstack((X[:, 0], y)).T, columns=['start_neighborhood',
                                                   'end_neighborhood'])

class SimpleFlassifiler():
    def __init__(self):
        return
    def fit(self, X, y):
        (self.lookup_df, self.lookup_dict, 
             self.labels) = dumb_fit(X, y)
        
    def score(self, X, y_true):
        y_preds = predict(self.lookup_dict, X)
        return log_loss(y_true, y_preds, labels=self.labels)

    def get_params(self, deep):
        return {}
    
def predict(lookup_dict, X):
    # return np.concatenate([lookup_dict[x[0]] for x in X])
    array_size = len(list(lookup_dict.values())[0])
    return np.concatenate([np.reshape(lookup_dict[x[0]], (1, array_size)) 
           for x in X])
```


```python
len(list(lookup.values())[0])
```




    54




```python
# Make data from mdf
# (where mdf here just contains the ['start_neighborhood',
#                                    'end_neighborhood'] )
X, y = mdf[['start_neighborhood']].values, np.array(mdf['end_neighborhood'].tolist())
clf = SimpleFlassifiler()
clf.fit(X, y)
#model = dumb_fit(X, y)
#X.shape, y.shape
clf.score(X, y)
```




    29.131042558171746




```python
#clf.score(X, y)
#(X[0][0])
#'Midtown East' in 
#print(list(clf.lookup_dict.keys()))
# y_preds = predict(clf.lookup_dict, X)
```


```python
clf.score(X, y)
```




    29.131042558171746




```python
#y_preds.shape, X.shape
array_size = len(clf.lookup_dict)
predarr = np.concatenate([np.reshape(clf.lookup_dict[x[0]], (1, array_size)) 
           for x in X])
```


```python
X[:5], y[:5]
```




    (array([['Midtown East'],
            ['Midtown East'],
            ['Midtown East'],
            ['Midtown East'],
            ['Midtown East']], dtype=object),
     array(['Stuyvesant Town', 'Stuyvesant Town', 'Stuyvesant Town',
            'Stuyvesant Town', 'Stuyvesant Town'], dtype='<U35'))




```python
import ipdb
```


```python

from sklearn.model_selection import cross_val_score
# clf = svm.SVC(kernel='linear', C=1)
clf = SimpleFlassifiler()
with ipdb.launch_ipdb_on_exception():
    scores = cross_val_score(clf, X, y, cv=5)

```


```python
scores
```




    array([29.03426394, 25.61716199, 29.19083979, 28.312853  , 22.04601817])


