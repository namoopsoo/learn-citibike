
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


    100%|██████████| 1/1 [00:59<00:00, 59.87s/it]
      0%|          | 0/1 [00:00<?, ?it/s]

    /opt/program/artifacts/2020-10-24T185245Z/train.libsvm


    100%|██████████| 1/1 [00:08<00:00,  8.93s/it]


    Done  /opt/program/artifacts/2020-10-24T185245Z/test.libsvm
    {'pmem': '17.9', 'rss': '0.348 GiB'}
    CPU times: user 40.5 s, sys: 9.02 s, total: 49.5 s
    Wall time: 1min 11s



```python
# Ok nice worked w/o kernel dying this time! 

# I did forget to label that as "v2" however.
# Doing manually to prevent confusion...

# (pandars3) $ cd artifacts/2020-10-24T185245Z/
# (pandars3) $ mv train.libsvm train.v2.libsvm
# (pandars3) $ mv test.libsvm test.v2.libsvm
```

#### Try the validation again 
Per https://github.com/namoopsoo/learn-citibike/blob/master/docs/common_tasks.md#validating-model-predictions


```python
print(train_loc, test_loc, workdir)
model = bundle['model_bundle']['bundle']['xgb_model']

dtest = xgb.DMatrix(f'{test_loc}?format=libsvm')
actuals = dtest.get_label()
```

    /opt/program/artifacts/2020-10-24T185245Z/train.libsvm /opt/program/artifacts/2020-10-24T185245Z/test.libsvm /opt/program/artifacts/2020-10-24T185245Z
    [20:19:41] 105427x85 matrix with 476566 entries loaded from /opt/program/artifacts/2020-10-24T185245Z/test.libsvm?format=libsvm



```python
from sklearn.metrics import log_loss, accuracy_score, balanced_accuracy_score

```


```python
# Can get probabilities and class predictions
y_prob_vec = model.predict(dtest)
predictions = np.argmax(y_prob_vec, axis=1)

logloss = log_loss(actuals, y_prob_vec, labels=list(range(54)))

acc = accuracy_score(actuals, predictions)
balanced_acc = balanced_accuracy_score(actuals, predictions)
```


```python
# acc # 0.11373746763163137
acc, balanced_acc
```




    (0.11373746763163137, 0.10922338775384331)




```python
# K area too ... 
import fresh.metrics as fm
correct_kth, topk, karea = fm.kth_area(actuals, y_prob_vec, num_classes=54)
```

    correct_kth_counts  {0: 11991, 1: 8324, 2: 6696, 3: 6016, 4: 5086, 5: 4521, 6: 4105, 7: 3522, 8: 3198, 9: 3128, 10: 2917, 11: 2770, 12: 2474, 13: 2376, 14: 2182, 15: 2050, 16: 1938, 17: 1873, 18: 1864, 19: 1793, 20: 1696, 21: 1646, 22: 1624, 23: 1489, 24: 1472, 25: 1348, 26: 1298, 27: 1273, 28: 1249, 29: 1182, 30: 1106, 31: 1022, 32: 1027, 33: 969, 34: 869, 35: 894, 36: 817, 37: 774, 38: 602, 39: 558, 40: 544, 41: 469, 42: 437, 43: 328, 44: 333, 45: 313, 46: 290, 47: 245, 48: 196, 49: 175, 50: 137, 51: 80, 52: 92, 53: 49}
    topk [0.         0.00210625 0.00356838 0.00474455 0.00580128 0.00669464
     0.00748877 0.00820982 0.00882847 0.00939021 0.00993965 0.01045203
     0.01093859 0.01137315 0.0117905  0.01217377 0.01253386 0.01287428
     0.01320327 0.01353069 0.01384563 0.01414354 0.01443267 0.01471792
     0.01497947 0.01523803 0.01547481 0.01570281 0.01592641 0.0161458
     0.01635343 0.0165477  0.01672721 0.01690761 0.01707782 0.01723046
     0.01738749 0.017531   0.01766696 0.0177727  0.01787071 0.01796627
     0.01804865 0.01812541 0.01818302 0.01824151 0.01829649 0.01834743
     0.01839047 0.0184249  0.01845563 0.0184797  0.01849375 0.01850991]



```python
acc, balanced_acc, karea
```




    (0.11373746763163137, 0.10922338775384331, 0.7532854926122305)




```python
# Ok, comparing to what I had in the bundle ... the below looks comparable.

bundle['model_bundle']['bundle']['validation_metrics']
```




    {'accuracy': 0.12171455130090014,
     'balanced_accuracy': 0.10451301995291779,
     'confusion': array([[415,  64,   4, ...,   0, 103,  69],
            [ 56, 541,   4, ...,   0, 130,  27],
            [ 23,  10, 136, ...,   0,  16, 130],
            ...,
            [  2,   0,   2, ...,   1,   3,  36],
            [151, 222,   3, ...,   0, 260,  35],
            [ 84,  25,  46, ...,   0,  29, 861]]),
     'logloss': 3.4335361255637977,
     'test': '/home/ec2-user/SageMaker/learn-citibike/artifacts/2020-07-08T143732Z/test.libsvm',
     'karea': 0.760827309330065}




```python
# So Next step would be to do a train session 
# like  per https://github.com/namoopsoo/learn-citibike/blob/master/notes/2020-07-03-aws.md#train 

print(fu.get_my_memory())

```

    {'pmem': '24.2', 'rss': '0.472 GiB'}



```python
# Need to make a dtrain/dtest that is v3 ..!! 
# ##dtrain = xgb.DMatrix(f'{train_loc}?format=libsvm')
import fresh.preproc.v3 as pv3
print(fu.get_my_memory())
```

    {'pmem': '24.3', 'rss': '0.473 GiB'}



```python
# Read X_train, y_train from my joblib dump from earlier...
data_bundle = joblib.load(
    '/opt/program/artifacts/2020-10-24T185245Z/train_test_data_bundle.joblib')
data_bundle.keys()
```




    dict_keys(['notebook', 'features', 'X_train', 'X_test', 'y_train', 'y_test', 'Xids_train', 'Xids_test', 'source_data', 'stations', 'timestamp'])




```python
features
```




    ['start_neighborhood',
     'start station id',
     'gender',
     'time_of_day',
     'usertype',
     'weekday',
     'age_bin']




```python
# Ok re-shape..
X_train = data_bundle['X_train'][:, [0, 2, 3, 4, 5, 6]]
X_test = data_bundle['X_test'][:, [0, 2, 3, 4, 5, 6]]
y_train, y_test, Xids_train, Xids_test = (
        data_bundle['y_train'], data_bundle['y_test'], 
        data_bundle['Xids_train'], data_bundle['Xids_test'])
```


```python
reload(pv3)
```




    <module 'fresh.preproc.v3' from '/opt/program/fresh/preproc/v3.py'>




```python
# hmm this is weird, xgboost is fine w/ nan 
# but the dump to svmlight doesnt like that the age column has `nan` ..
# "ValueError: Input contains NaN, infinity or a value too large for dtype('float64')."
# Guess I have to encode the nan as `-1` 
# Because according to https://github.com/scikit-learn/scikit-learn/issues/7110  , 
# svmlight sparseness doesnt refer to missing data but sparse items are "0".
```


```python

np.nan_to_num(np.concatenate([
#np.array([9, np.nan, 9]),
    X_train[1:10, 5],
]
)

, nan=-1#, #nan=-1
             )
#type (X_train[1:2, 5][0]), type(np.nan)
# import ipdb; ipdb.runcall(np.nan_to_num, X_train[0:3, 5])
# np.nan_to_num(X_train[0:3, 5].copy())
#.dtype
np.nan_to_num(X_train[0:3, 5].astype(np.float64), nan=-1)

```




    array([ 2., -1.,  2.])




```python
print("Creating a numpy array from a mixed type DataFrame" 
      "can create an 'object' numpy array dtype:")
A = np.array([1., 2., 3., np.nan]); print('A:', A, A.dtype)
B = pd.DataFrame([[1., 2., 3., np.nan,],  [1, 2, 3, '4']]
                  ).to_numpy();  print('B:', B, B.dtype, '\n')

print('Converting vanilla A is fine:\n', np.nan_to_num(A, nan=-99), '\n')
print('But not B:\n', np.nan_to_num(B, nan=-99), '\n')
print('Not even this slice of B, \nB[0, :] : ', B[0, :])
print(np.nan_to_num(B[0, :], nan=-99), '\n')

print('The astype(np.float64) does the trick here:\n', 
      np.nan_to_num(B[0, :].astype(np.float64), nan=-99), '\n\n')
```

    Creating a numpy array from a mixed type DataFrame can create an 'object' numpy array dtype:
    A: [ 1.  2.  3. nan] float64
    B: [[1.0 2.0 3.0 nan]
     [1.0 2.0 3.0 '4']] object 
    
    Converting vanilla A is fine:
     [  1.   2.   3. -99.] 
    
    But not B:
     [[1.0 2.0 3.0 nan]
     [1.0 2.0 3.0 '4']] 
    
    Not even this slice of B, 
    B[0, :] :  [1.0 2.0 3.0 nan]
    [1.0 2.0 3.0 nan] 
    
    The astype(np.float64) does the trick here:
     [  1.   2.   3. -99.] 
    
    



```python
# issubclass(type(np.nan), np.inexact)
reload(pv3)
```




    <module 'fresh.preproc.v3' from '/opt/program/fresh/preproc/v3.py'>




```python
%%time
print('Creating v3 proc bundle ')
proc_bundle, train_loc = pv3.preprocess(
        X_train, y_train, neighborhoods, #proc_bundle=proc_bundle,
        workdir=workdir,
        dataset_name='train.v3')
print('pv3 train_loc', train_loc)
```

    Creating v3 proc bundle 


    100%|██████████| 1/1 [00:29<00:00, 29.79s/it]

    pv3 train_loc /opt/program/artifacts/2020-10-24T185245Z/train.v3.libsvm
    CPU times: user 12.8 s, sys: 14.2 s, total: 27 s
    Wall time: 30.8 s


    



```python
test_loc = pv3.preprocess(
        X_test, y_test, neighborhoods, proc_bundle=proc_bundle,
        workdir=workdir,
        dataset_name='test.v3')
print('Done ', test_loc)

print(fu.get_my_memory())
```

    100%|██████████| 1/1 [00:10<00:00, 10.49s/it]


    Done  /opt/program/artifacts/2020-10-24T185245Z/test.v3.libsvm
    {'pmem': '14.9', 'rss': '0.291 GiB'}


### 2020-10-25

#### Ok going to try to use this now



```python
# Using the same params from last model
num_round = bundle['model_bundle']['bundle']['num_round']
params = bundle['model_bundle']['bundle']['input_params']
print(train_loc, test_loc)
```

    /opt/program/artifacts/2020-10-24T185245Z/train.v3.libsvm /opt/program/artifacts/2020-10-24T185245Z/test.v3.libsvm



```python
%%time
print(fu.get_my_memory())
dtrain = xgb.DMatrix(f'{train_loc}?format=libsvm')
print(fu.get_my_memory())
# dtest = xgb.DMatrix(f'{test_loc}?format=libsvm')

watchlist = [(dtrain, 'train'), 
             #(dtest, 'test')
            ]
num_round = 100
fu.log(workdir, 'Start xgb.train')
xgb_model = xgb.train(params, dtrain, num_round, watchlist)
```

    {'pmem': '15.0', 'rss': '0.292 GiB'}
    [17:31:38] 316281x86 matrix with 1690798 entries loaded from /opt/program/artifacts/2020-10-24T185245Z/train.v3.libsvm?format=libsvm
    {'pmem': '17.7', 'rss': '0.345 GiB'}
    [0]	train-merror:0.893474
    [1]	train-merror:0.888352
    [2]	train-merror:0.885093
    [3]	train-merror:0.883635
    [4]	train-merror:0.88244
    [5]	train-merror:0.881748
    [6]	train-merror:0.881213
    [7]	train-merror:0.880666
    [8]	train-merror:0.880805
    [9]	train-merror:0.880508
    [10]	train-merror:0.879949
    [11]	train-merror:0.879727
    [12]	train-merror:0.879092
    [13]	train-merror:0.878848
    [14]	train-merror:0.87864
    [15]	train-merror:0.878437
    [16]	train-merror:0.878434
    [17]	train-merror:0.877988
    [18]	train-merror:0.877735
    [19]	train-merror:0.877637
    [20]	train-merror:0.877482
    [21]	train-merror:0.87722
    [22]	train-merror:0.876885
    [23]	train-merror:0.876663
    [24]	train-merror:0.876578
    [25]	train-merror:0.876256
    [26]	train-merror:0.876151
    [27]	train-merror:0.875822
    [28]	train-merror:0.875668
    [29]	train-merror:0.875535
    [30]	train-merror:0.87556
    [31]	train-merror:0.875642
    [32]	train-merror:0.875358
    [33]	train-merror:0.875244
    [34]	train-merror:0.874953
    [35]	train-merror:0.874801
    [36]	train-merror:0.874747
    [37]	train-merror:0.874747
    [38]	train-merror:0.874627
    [39]	train-merror:0.874551
    [40]	train-merror:0.874242
    [41]	train-merror:0.874242
    [42]	train-merror:0.874096
    [43]	train-merror:0.873985
    [44]	train-merror:0.873755
    [45]	train-merror:0.873831
    [46]	train-merror:0.873691
    [47]	train-merror:0.873701
    [48]	train-merror:0.87354
    [49]	train-merror:0.873457
    [50]	train-merror:0.873461
    [51]	train-merror:0.873372
    [52]	train-merror:0.873375
    [53]	train-merror:0.873107
    [54]	train-merror:0.873189
    [55]	train-merror:0.873261
    [56]	train-merror:0.87316
    [57]	train-merror:0.873144
    [58]	train-merror:0.873005
    [59]	train-merror:0.872993
    [60]	train-merror:0.872967
    [61]	train-merror:0.872762
    [62]	train-merror:0.872645
    [63]	train-merror:0.872702
    [64]	train-merror:0.872626
    [65]	train-merror:0.872322
    [66]	train-merror:0.872272
    [67]	train-merror:0.872117
    [68]	train-merror:0.872218
    [69]	train-merror:0.872259
    [70]	train-merror:0.872209
    [71]	train-merror:0.872259
    [72]	train-merror:0.872209
    [73]	train-merror:0.872228
    [74]	train-merror:0.872088
    [75]	train-merror:0.872104
    [76]	train-merror:0.871962
    [77]	train-merror:0.871798
    [78]	train-merror:0.871801
    [79]	train-merror:0.871785
    [80]	train-merror:0.871769
    [81]	train-merror:0.871598
    [82]	train-merror:0.871554
    [83]	train-merror:0.871485
    [84]	train-merror:0.871387
    [85]	train-merror:0.871462
    [86]	train-merror:0.87119
    [87]	train-merror:0.871292
    [88]	train-merror:0.871137
    [89]	train-merror:0.871156
    [90]	train-merror:0.871045
    [91]	train-merror:0.871115
    [92]	train-merror:0.871165
    [93]	train-merror:0.871023
    [94]	train-merror:0.871096
    [95]	train-merror:0.870855
    [96]	train-merror:0.870843
    [97]	train-merror:0.870896
    [98]	train-merror:0.870925
    [99]	train-merror:0.870852
    CPU times: user 19min 34s, sys: 50 s, total: 20min 24s
    Wall time: 5min 7s



```python
# First save this one in joblib

bundle_loc = f'{workdir}/1_model_bundle.joblib'
print('Saving to ', bundle_loc)

joblib.dump({
    
    'notebook': '2020-10-23-quick-new-v3-proc-bundle.ipynb',
    'machine':{'type': 'laptop', 'what': 'MacBook Pro (Retina, Mid 2012)'},
    'workdir': workdir,
    'xgb_model': xgb_model,
    'features': ['start_neighborhood', 'gender', 'time_of_day', 'usertype', 'weekday', 
                    'age_bin'],
    'git_hash': 'aa1ef10',
    'train': {'train_loc': train_loc,
             'train_error': 0.870852, 
             'train_acc': 1 - 0.870852
             },

    'walltime': '5min 7s',
    'primary_dataset': '2013-07 - Citi Bike trip data.csv',
    'input_params': params,
    'num_round': num_round,
    'proc_bundle': {
        'bundle': proc_bundle,
        'version': 'v3'
    },
    'model_id': f'{workdir}_1',
    'data_bundle': {
        'loc': '/opt/program/artifacts/2020-10-24T185245Z/train_test_data_bundle.joblib',   
    },
    'timestamp': fu.utc_ts(),


}, bundle_loc)
fu.log(workdir, f'wrote bundle {bundle_loc}')
```

    Saving to  /opt/program/artifacts/2020-10-24T185245Z/1_model_bundle.joblib



```python
# Delete stuff to make space and Validate...
print(fu.get_my_memory())
del dtrain
print(fu.get_my_memory())
del X_train, X_test, y_train, y_test, Xids_train, Xids_test
print(fu.get_my_memory())
print('test_loc', test_loc)
dtest = xgb.DMatrix(f'{test_loc}?format=libsvm')
print(fu.get_my_memory())
```

    {'pmem': '33.0', 'rss': '0.642 GiB'}
    {'pmem': '33.0', 'rss': '0.642 GiB'}
    {'pmem': '33.0', 'rss': '0.642 GiB'}
    test_loc /opt/program/artifacts/2020-10-24T185245Z/test.v3.libsvm
    [18:18:32] 105427x86 matrix with 563643 entries loaded from /opt/program/artifacts/2020-10-24T185245Z/test.v3.libsvm?format=libsvm
    {'pmem': '33.0', 'rss': '0.642 GiB'}



```python
del data_bundle
print(fu.get_my_memory())
```

    {'pmem': '33.0', 'rss': '0.642 GiB'}



```python
%%time
actuals = dtest.get_label()
y_prob_vec = xgb_model.predict(dtest)
predictions = np.argmax(y_prob_vec, axis=1)

acc = accuracy_score(actuals, predictions)
balanced_acc = balanced_accuracy_score(actuals, predictions)
print('acc', acc, 'balanced_acc', balanced_acc)
logloss = log_loss(actuals, y_prob_vec, labels=list(range(54)))
print('logloss', logloss)

correct_kth, topk, karea = fm.kth_area(actuals, y_prob_vec, num_classes=54)
print('karea', karea)
```

    acc 0.1209747028749751 balanced_acc 0.10549608878985327
    logloss 3.4214785990026457
    correct_kth_counts  {0: 12754, 1: 8727, 2: 6982, 3: 5926, 4: 5262, 5: 4725, 6: 4034, 7: 3699, 8: 3428, 9: 3163, 10: 2907, 11: 2675, 12: 2509, 13: 2216, 14: 2152, 15: 2069, 16: 1874, 17: 1866, 18: 1775, 19: 1641, 20: 1619, 21: 1558, 22: 1490, 23: 1395, 24: 1349, 25: 1257, 26: 1178, 27: 1162, 28: 1149, 29: 1124, 30: 1074, 31: 1012, 32: 919, 33: 890, 34: 841, 35: 722, 36: 713, 37: 751, 38: 586, 39: 559, 40: 502, 41: 452, 42: 378, 43: 389, 44: 347, 45: 305, 46: 270, 47: 248, 48: 207, 49: 201, 50: 168, 51: 93, 52: 79, 53: 56}
    topk [0.         0.00224027 0.00377319 0.0049996  0.00604051 0.0069648
     0.00779476 0.00850334 0.00915308 0.00975521 0.0103108  0.01082142
     0.0112913  0.01173201 0.01212125 0.01249926 0.01286268 0.01319186
     0.01351962 0.01383141 0.01411965 0.01440403 0.0146777  0.01493942
     0.01518446 0.01542141 0.01564221 0.01584913 0.01605324 0.01625506
     0.01645249 0.01664114 0.0168189  0.01698033 0.01713666 0.01728438
     0.01741121 0.01753645 0.01766836 0.01777129 0.01786948 0.01795766
     0.01803705 0.01810345 0.01817178 0.01823273 0.01828631 0.01833373
     0.01837729 0.01841365 0.01844896 0.01847847 0.01849481 0.01850868]
    karea 0.7613679677951641
    CPU times: user 1min 27s, sys: 804 ms, total: 1min 28s
    Wall time: 29.7 s



```python
# Second save to joblib , w/ validation metrics too

bundle_loc = f'{workdir}/1_model_bundle.joblib'
print('Saving to ', bundle_loc)

joblib.dump({
    
    'notebook': '2020-10-23-quick-new-v3-proc-bundle.ipynb',
    'machine':{'type': 'laptop', 'what': 'MacBook Pro (Retina, Mid 2012)'},
    'workdir': workdir,
    'xgb_model': xgb_model,
    'features': ['start_neighborhood', 'gender', 'time_of_day', 'usertype', 'weekday', 
                    'age_bin'],
    'git_hash': 'aa1ef10',
    'train': {'train_loc': train_loc,
             'train_error': 0.870852, 
             'train_acc': 1 - 0.870852
             },

    'walltime': '5min 7s',
    'primary_dataset': '2013-07 - Citi Bike trip data.csv',
    'input_params': params,
    'num_round': num_round,
    'proc_bundle': {
        'bundle': proc_bundle,
        'version': 'v3'
    },
    'model_id': f'{workdir}_1',
    'data_bundle': {
        'loc': '/opt/program/artifacts/2020-10-24T185245Z/train_test_data_bundle.joblib',   
    },
    'timestamp': fu.utc_ts(),
    'validation_metrics': {
        'test_loc': test_loc,
        'acc': acc,
        'balanced_acc': balanced_acc,
        'karea': karea,
        'logloss': logloss
    },
    'docker_image': 'citibike-learn:0.9'
}, bundle_loc)
fu.log(workdir, f'wrote bundle {bundle_loc}')
```

    Saving to  /opt/program/artifacts/2020-10-24T185245Z/1_model_bundle.joblib



```python
# oops overwrote the other one, but that's probably fine. Just added additional stuff.
```
