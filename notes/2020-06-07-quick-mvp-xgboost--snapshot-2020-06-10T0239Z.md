

```python
import pandas as pd
import xgboost
import datetime; import pytz
import matplotlib as plt
from scipy.special import softmax
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split # (*arrays, **options)
import numpy as np
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder
```


```python

def utc_ts():
    return datetime.datetime.utcnow(
        ).replace(tzinfo=pytz.UTC).strftime('%Y-%m-%dT%H%M%SZ')
utc_ts()
```




    '2020-06-08T142456Z'



#### The data



```python
datadir = '/opt/data'
localdir = '/opt/program'
tripsdf = pd.read_csv(f'{datadir}/2013-07 - Citi Bike trip data.csv')
stationsdf = pd.read_csv(f'{localdir}/datas/stations/stations-2018-12-04-c.csv',
                        index_col=0)
```


```python
def prepare_data(tripsdf, stationsdf):
    
    # Step 1, merge w/ stationsdf to get neighborhood data
    mdf = tripsdf[['start station name', 'end station name', 'gender']
            ].merge(stationsdf[['station_name', 'neighborhood']], 
                    left_on='start station name',
                    right_on='station_name'
                   ).rename(columns={'neighborhood': 'start_neighborhood'}
                           ).merge(stationsdf[['station_name', 'neighborhood']],
                                  left_on='end station name',
                                   right_on='station_name'
                                  ).rename(columns={'neighborhood': 'end_neighborhood'})
    
    neighborhoods = sorted(stationsdf.neighborhood.unique().tolist())
    genders = [0, 1, 2]
    
    X, y = (mdf[['start_neighborhood', 'gender']].values, 
            np.array(mdf['end_neighborhood'].tolist()))
    return X, y
    
def preprocess(X, y, labeled):
    # Initially assuming labeled=True
    labeled = True
    
    enc = OneHotEncoder(handle_unknown='error', 
                        categories=[neighborhoods, genders])
    enc.fit(X)
    X_transformed = enc.transform(X)
    
    le = preprocessing.LabelEncoder()
    le.fit(neighborhoods)
    
    return X, X_transformed, y, enc, le
    
class FooFlassifier():
    def __init__(self, stationsdf):
        self.stationsdf = stationsdf
    def fit(self, X, y):
        # preproc
        X, X_transformed, y, enc, le = preprocess(X, y, True)
        
    def score(self, X, y):
        return .9
    def get_params(self, deep):
        return {}
    
X, y = mdf[['start_neighborhood', 'gender']].values, np.array(mdf['end_neighborhood'].tolist())

clf = FooFlassifier(stationsdf) 
```


```python
tripsdf.iloc[0]
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
(X, X_transformed, y, enc) = preprocess(tripsdf, stationsdf, labeled=True)
```


```python
X_transformed[0].toarray()
```




    array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]])




```python
X_transformed.shape, X.shape, y.shape
```




    ((843416, 78), (843416, 2), (843416,))




```python
y[:5]
```




    array(['Stuyvesant Town', 'Stuyvesant Town', 'Stuyvesant Town',
           'Stuyvesant Town', 'Stuyvesant Town'], dtype='<U35')




```python
from sklearn.preprocessing import LabelEncoder
# Get data..
neighborhoods = sorted(stationsdf.neighborhood.unique().tolist())
le = LabelEncoder()
le.fit(neighborhoods)
y_enc = le.transform(y)
#Xdm = xgboost.DMatrix(X_transformed[:5,:])
#from xgboost import XGBClassifier
clf = xgboost.XGBClassifier()
```


```python
len(neighborhoods)
```




    75




```python
X_transformed.shape, y_enc.shape
```




    ((843416, 78), (843416,))




```python
%time clf.fit(X_transformed, y_enc)
```

    CPU times: user 13min 58s, sys: 2.19 s, total: 14min
    Wall time: 13min 59s





    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=1, gamma=0,
                  learning_rate=0.1, max_delta_step=0, max_depth=3,
                  min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
                  nthread=None, objective='multi:softprob', random_state=0,
                  reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                  silent=None, subsample=1, verbosity=1)




```python
clf.predict_proba(X_transformed[:2])
```




    array([[0.01552367, 0.01610189, 0.00264415, 0.00248007, 0.01490377,
            0.00548938, 0.00194296, 0.02104487, 0.08323692, 0.00932526,
            0.01069923, 0.00389162, 0.0011591 , 0.0095946 , 0.00404946,
            0.0107293 , 0.01842718, 0.0157086 , 0.01104296, 0.00274902,
            0.02098426, 0.0144077 , 0.02674809, 0.0424823 , 0.00737197,
            0.0533537 , 0.00576095, 0.01814625, 0.00236372, 0.02293371,
            0.01015841, 0.00756694, 0.06421325, 0.1442597 , 0.08239184,
            0.03230955, 0.00267815, 0.00657854, 0.00776298, 0.00174899,
            0.00284085, 0.00655629, 0.00576877, 0.00975472, 0.00180184,
            0.0414222 , 0.02034637, 0.0035512 , 0.00375667, 0.02123439,
            0.0110102 , 0.00067188, 0.02583351, 0.01048607],
           [0.01552367, 0.01610189, 0.00264415, 0.00248007, 0.01490377,
            0.00548938, 0.00194296, 0.02104487, 0.08323692, 0.00932526,
            0.01069923, 0.00389162, 0.0011591 , 0.0095946 , 0.00404946,
            0.0107293 , 0.01842718, 0.0157086 , 0.01104296, 0.00274902,
            0.02098426, 0.0144077 , 0.02674809, 0.0424823 , 0.00737197,
            0.0533537 , 0.00576095, 0.01814625, 0.00236372, 0.02293371,
            0.01015841, 0.00756694, 0.06421325, 0.1442597 , 0.08239184,
            0.03230955, 0.00267815, 0.00657854, 0.00776298, 0.00174899,
            0.00284085, 0.00655629, 0.00576877, 0.00975472, 0.00180184,
            0.0414222 , 0.02034637, 0.0035512 , 0.00375667, 0.02123439,
            0.0110102 , 0.00067188, 0.02583351, 0.01048607]], dtype=float32)




```python
neighborhoods = ['Midtown East', 'Stuyvesant Town',]; gender = [0,1,2]
enc = OneHotEncoder(handle_unknown='ignore', categories=[neighborhoods, gender])
X = np.array([['Midtown East', 1],
             ['Midtown East',2],
             ['Stuyvesant Town',0]])
X2 = np.array([['foo', 1],
             ['Midtown East',2],
             ['Stuyvesant Town',0]])
enc.fit(X)

```




    OneHotEncoder(categories=[['Midtown East', 'Stuyvesant Town'], [0, 1, 2]],
                  drop=None, dtype=<class 'numpy.float64'>, handle_unknown='ignore',
                  sparse=True)




```python
enc.transform(X2).toarray()
```




    array([[0., 0., 0., 1., 0.],
           [1., 0., 0., 0., 1.],
           [0., 1., 1., 0., 0.]])




```python
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
le = preprocessing.LabelEncoder()
neigh
```
