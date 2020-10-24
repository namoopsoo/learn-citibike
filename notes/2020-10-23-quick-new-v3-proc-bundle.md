
- I tried making the new proc bundle in [yesterdays notebook](http://127.0.0.1:8889/notebooks/2020-10-22-features-v3.ipynb) but kernel crashed and burned haha on the `pv2.preprocess` step.

- But to be fair, originally, in [this](https://github.com/namoopsoo/learn-citibike/blob/master/notes/2020-07-03-aws.md) notebook which I am re-doing, I had deleted `traindf` from memory before `pv2.preprocess`, clearing up roughly `0.3GiB`. 
- Going to retry and with the same `del` step!


```python
import datetime
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pytz
import xgboost as xgb

from collections import Counter
from importlib import reload
from joblib import dump, load
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
from xgboost import XGBClassifier
import fresh.predict_utils as fpu
import fresh.preproc.v2 as pv2
import fresh.utils as fu
```


```python
fu.get_my_memory()
```




    {'pmem': '7.5', 'rss': '0.147 GiB'}




```python
localdir = '/opt/program'  # laptop docker 
# localdir = '/home/ec2-user/SageMaker/learn-citibike'  # sagemaker
datadir = '/opt/data'

workdir = fu.make_work_dir(localdir); print(workdir)
fu.log(workdir, 'new workdir')
```

    /opt/program/artifacts/2020-10-24T185245Z



```python
bundle = fpu.load_bundle_in_docker()
fu.get_my_memory()
```

    Loading from bundle_loc /opt/ml/model/all_bundle_with_stationsdf.joblib





    {'pmem': '8.0', 'rss': '0.156 GiB'}



#### how to identify the data being split?
- is the (station, starttime) unique? 
- Maybe not but oh actually the (bikeid, starttime) should be unique.
- Also mini idea haha one can use the data to create a "busy-ness" feature. Maybe that does something.


```python
# Wow fascinating, 25% of these trips started at the same time
tripsdf.shape, tripsdf.drop_duplicates(subset=['starttime']).shape
```


```python
# Wow indeed, 0.4% of trips actually started simultaneously at same stations! 
tripsdf.shape, tripsdf.drop_duplicates(subset=['starttime', 'start station id']).shape
```


```python
# Ok cool no weird duplicate bike ids... so this can be a nice unique row id.
tripsdf.shape, tripsdf.drop_duplicates(subset=['starttime', 'bikeid']).shape
```


```python
a, b, c, d, e, f = train_test_split(np.arange(1, 10),
                                    np.arange(11, 20),
                                    np.arange(21, 30))
```


```python
# So based on this, I can just pass an ids array and we will be golden.
print(a, b)
print(c, d)
print(e, f)
```


```python
tripsdf.iloc[0]
```


```python
def _starttime_clean_1(x):
    return datetime.datetime.strptime(x, 
                                     '%Y-%m-%d %H:%M:%S'
                                    ).strftime('%Y-%m-%dT%H%M%S')

def _starttime_clean_2(x):
    parts = x.split(' ')
    return f'{parts[0]}T{"".join(parts[1].split(":"))}'

_starttime_clean_2('2013-07-01 00:00:00')
```


```python
%%time
print(_starttime_clean_2('2013-07-01 00:00:00'))
```


```python
%%time
print(_starttime_clean_1('2013-07-01 00:00:00'))
```


```python
%%time
print(fu.get_my_memory())
tripsdf = pd.read_csv(f'{datadir}/2013-07 - Citi Bike trip data.csv'
                     )#.sample(frac=0.017, random_state=42)
stationsdf = pd.read_csv(f'{localdir}/datas/stations/stations-2018-12-04-c.csv',
                        index_col=0)
print('read tripsdf, stationsdf')
print(fu.get_my_memory())
```

    {'pmem': '8.0', 'rss': '0.156 GiB'}
    read tripsdf, stationsdf
    {'pmem': '18.9', 'rss': '0.368 GiB'}
    CPU times: user 2.13 s, sys: 65.5 ms, total: 2.2 s
    Wall time: 3.26 s



```python
%%time
print(fu.get_my_memory())
reload(fu)
Xids, features, X, y = fu.prepare_data(tripsdf, stationsdf, labelled=True)
print(fu.get_my_memory())
```

    {'pmem': '18.9', 'rss': '0.368 GiB'}
    {'pmem': '35.5', 'rss': '0.692 GiB'}
    CPU times: user 7min 46s, sys: 2.46 s, total: 7min 49s
    Wall time: 7min 49s



```python
fu.utc_ts()
```




    '2020-10-24T191821Z'




```python
%%time
# Try to save to disk w/ joblib... 
print('Try saving to disk w/ joblib', workdir)
out_loc = f'{workdir}/data_bundle.joblib'
joblib.dump({'notebook': '2020-10-23-quick-new-v3-proc-bundle',
             'features': features,
             'Xids': Xids,
             'X': X,
             'y': y,
             'source_dataset': '2020-10-23-quick-new-v3-proc-bundle',
             'stations': 'stations-2018-12-04-c.csv',
             'timestamp': fu.utc_ts()}, out_loc)
```

    Try saving to disk w/ joblib /opt/program/artifacts/2020-10-24T185245Z
    CPU times: user 1.25 s, sys: 207 ms, total: 1.46 s
    Wall time: 3.47 s





    ['/opt/program/artifacts/2020-10-24T185245Z/data_bundle.joblib']




```python
!ls -lah /opt/program/artifacts/2020-10-24T185245Z/data_bundle.joblib
```

    -rw-r--r-- 1 root root 157M Oct 24 19:18 /opt/program/artifacts/2020-10-24T185245Z/data_bundle.joblib



```python
print(fu.get_my_memory())
del tripsdf, stationsdf
print(fu.get_my_memory())
```

    {'pmem': '36.0', 'rss': '0.701 GiB'}
    {'pmem': '29.1', 'rss': '0.567 GiB'}



```python
%%time
# X.shape, y.shape, fu.get_proportions(y)
print(fu.get_my_memory())

# Ideally the balancing is best done after train/test split, 
# But that is what I had earlier here , https://github.com/namoopsoo/learn-citibike/blob/master/notes/2020-07-03-aws.md 
# So it is more important to re-create what I had for now.
X_balanced, y_balanced, Xids_balanced = fu.balance_dataset(
                X=X, y=y, Xids=Xids, shrinkage=.5)
X_balanced.shape, y_balanced.shape, fu.get_proportions(y_balanced)

X, y = X_balanced, y_balanced # rename for simplicity..
# Xids = Xids_balanced
print(fu.get_my_memory())
```

    {'pmem': '29.1', 'rss': '0.567 GiB'}
    {'pmem': '26.3', 'rss': '0.513 GiB'}
    CPU times: user 4.94 s, sys: 159 ms, total: 5.1 s
    Wall time: 5.16 s



```python
%%time
# Try to save to disk w/ joblib... 
print('Try saving to disk w/ joblib', workdir)
out_loc = f'{workdir}/balanced_data_bundle.joblib'
joblib.dump({'notebook': '2020-10-23-quick-new-v3-proc-bundle',
             'features': features,
             'Xids': Xids,
             'X': X,
             'y': y,
             'pre_balance_data': '/opt/program/artifacts/2020-10-24T185245Z/data_bundle.joblib',
             'source_dataset': '2020-10-23-quick-new-v3-proc-bundle',
             'stations': 'stations-2018-12-04-c.csv',
             'timestamp': fu.utc_ts()}, out_loc)

```

    Try saving to disk w/ joblib /opt/program/artifacts/2020-10-24T185245Z
    CPU times: user 461 ms, sys: 92.6 ms, total: 554 ms
    Wall time: 1.61 s





    ['/opt/program/artifacts/2020-10-24T185245Z/balanced_data_bundle.joblib']




```python
print(fu.get_my_memory())

```

    {'pmem': '23.4', 'rss': '0.456 GiB'}



```python
print(fu.get_my_memory())
X_train, X_test, y_train, y_test, Xids_train, Xids_test = train_test_split(X, y, Xids)
print(fu.get_my_memory())

```

    {'pmem': '23.4', 'rss': '0.456 GiB'}
    {'pmem': '25.5', 'rss': '0.497 GiB'}



```python
print(fu.get_my_memory())
del X, y, Xids
print(fu.get_my_memory())
```

    {'pmem': '25.5', 'rss': '0.497 GiB'}
    {'pmem': '25.5', 'rss': '0.497 GiB'}



```python
X_train.shape, X_test.shape, y_train.shape, y_test.shape, Xids_train.shape, Xids_test.shape
```




    ((316281, 7), (105427, 7), (316281,), (105427,), (316281,), (105427,))




```python
# Finally.. Save the train/test split 

print('Try saving to disk w/ joblib', workdir)
out_loc = f'{workdir}/train_test_data_bundle.joblib'
joblib.dump({'notebook': '2020-10-23-quick-new-v3-proc-bundle',
             'features': features,
             'X_train': X_train, 
             'X_test': X_test, 
             'y_train': y_train, 
             'y_test': y_test, 
             'Xids_train': Xids_train, 
             'Xids_test': Xids_test,
             'source_data': [['original', '2013-07 - Citi Bike trip data.csv'],
                              ['prepared', 
                               '/opt/program/artifacts/2020-10-24T185245Z/data_bundle.joblib'],
                             ['balanced', 
                             '/opt/program/artifacts/2020-10-24T185245Z/balanced_data_bundle.joblib'],
                            ],

             'stations': 'stations-2018-12-04-c.csv',
             'timestamp': fu.utc_ts()}, out_loc)

```

    Try saving to disk w/ joblib /opt/program/artifacts/2020-10-24T185245Z





    ['/opt/program/artifacts/2020-10-24T185245Z/train_test_data_bundle.joblib']



#### Preprocess
Oh oops.. I tried to do pv2.preproces but got error 

```
ValueError: Found unknown categories [72, 79, 82, 83, ..., 540, 545, 546,..., 2023] in column 1 during transform
```
Because oops since I changed around X, I have to do some slicing so it matches what is expected!! 


```python
# Make X look like what pv2 expects...
# ['start_neighborhood', 'gender', 'time_of_day', 'usertype', 'weekday']
# [['Midtown East' 0 4 'Customer' 1]]

# Now we have this:
# X_train[0] # array(['Union Square', 497, 1, 3, 'Subscriber', 1, 2.0], dtype=object)

print(X_train[:5, [0, 2, 3, 4, 5]])
print('before remapping', X_train.shape)
X_train = X_train[:, [0, 2, 3, 4, 5]]
X_test = X_test[:, [0, 2, 3, 4, 5]]
print('after remapping', X_train.shape)
```

    [['Union Square' 1 3 'Subscriber' 1]
     ['Battery Park City' 0 3 'Customer' 0]
     ['West Village' 1 1 'Subscriber' 1]
     ['Chelsea' 1 1 'Subscriber' 1]
     ['Ukrainian Village' 1 4 'Subscriber' 1]]
    before remapping (316281, 7)
    after remapping (316281, 5)



```python
%%time
# Try to do preprocess again!
# Hopefully kernel will not die like yesterday 
#   in 2020-10-22-features-v3.ipynb

print(fu.get_my_memory())
proc_bundle = bundle['proc_bundle']['bundle']['proc_bundle']
neighborhoods = bundle['neighborhoods_bundle']['neighborhoods']
print(fu.get_my_memory())

print('Creating new train/test using existing proc bundle')
train_loc = pv2.preprocess(
        X_train, y_train, neighborhoods, proc_bundle=proc_bundle,
        workdir=workdir,
        dataset_name='train')
print(train_loc)

test_loc = pv2.preprocess(
        X_test, y_test, neighborhoods, proc_bundle=proc_bundle,
        workdir=workdir,
        dataset_name='test')
print('Done ', test_loc)

print(fu.get_my_memory())

```

      0%|          | 0/1 [00:00<?, ?it/s]

    {'pmem': '26.8', 'rss': '0.522 GiB'}
    {'pmem': '26.8', 'rss': '0.522 GiB'}
    Creating new train/test using existing proc bundle

