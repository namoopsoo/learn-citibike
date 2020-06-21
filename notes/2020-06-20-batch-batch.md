
#### redo w/ batch
- would like to re-do the "2020-06-20" notebook but w/ the batching approach, to see if any deterioration!!?? 
- That is, the batching approach I had used [here](https://github.com/namoopsoo/learn-citibike/blob/2020-revisit/notes/2020-06-14.md#trying-out-that-model-save)


```python
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
import datetime; import pytz
import matplotlib.pyplot as plt
from scipy.special import softmax
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split # (*arrays, **options)
import numpy as np
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from joblib import dump, load
import joblib
import os
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import fresh.utils as fu

from importlib import reload
from collections import Counter
from tqdm.notebook import tqdm
import fresh.preproc.v1 as pv1
```


```python
datadir = '/opt/data'
localdir = '/opt/program'


tripsdf = pd.read_csv(f'{datadir}/2013-07 - Citi Bike trip data.csv'
                     )#.sample(frac=0.017, random_state=42)
stationsdf = pd.read_csv(f'{localdir}/datas/stations/stations-2018-12-04-c.csv',
                        index_col=0)

```


```python
X, y, neighborhoods = fu.prepare_data(tripsdf, stationsdf)

# ... actually doing this part jsut to get those labels... 
X_train, X_test, y_train, y_test = train_test_split(X, y)
# preproc
(X_transformed,
     one_hot_enc, le,
     y_enc) = pv1.preprocess(X_train, y_train, # X[train_index]
                         neighborhoods)
labels = le.classes_

```


```python
def do_train(model, X, y, workdir):

    size = X.shape[0]
    while True:
        indices = np.random.choice(range(size), size=size, replace=False)
        parts = fu.get_partitions(indices, slice_size=10000)

        if len(Counter(y_enc[parts[0]])) == 54:
            break
        print('..shuffling..')
        
    prev_model_loc = None
    for i, part in enumerate(parts):
        model.fit(X[part], y[part], xgb_model=prev_model_loc)
        fu.log(workdir, f'({i}/{len(parts)}) Done fit', f'mem, ({fu.get_my_memory()})')

        prev_model_loc = f'{workdir}/model.xg'
        model.save_model(prev_model_loc)

    return model
```


```python
reload(fu);print(f'mem, ({fu.get_my_memory()})')
```

    mem, ({'pmem': '34.7', 'rss': '0.678 GiB'})



```python
%%time
workdir = fu.make_work_dir(); print(workdir)
fu.log(workdir, 'Starting, ', f'mem, ({fu.get_my_memory()})')

rng = np.random.RandomState(31337)

kf = KFold(n_splits=2, shuffle=True, random_state=rng)
for (i, (train_index, test_index)) in enumerate(kf.split(X)):    
    # preproc
    (X_transformed,
         one_hot_enc, le,
         y_enc) = pv1.preprocess(X[train_index], y[train_index], 
                             neighborhoods)
    
    xgb_model = xgb.XGBClassifier(objective='multi:softprob'
                                 )# .fit(X_transformed, y_enc, verbose=True)
    
    xgb_model = do_train(xgb_model, X_transformed, y_enc, workdir=workdir)
    fu.log(workdir, f'[{i}] Done fit.', f'mem, ({fu.get_my_memory()})')
    
    bundle_loc = f'{workdir}/bundle_{i}.joblib'
    joblib.dump({'model': xgb_model}, bundle_loc)
    #
    X_test_transformed = one_hot_enc.transform(X[test_index])
    actuals = le.transform(y[test_index]); len(actuals)
    
    predictions = xgb_model.predict(X_test_transformed)
    confusion = confusion_matrix(actuals, predictions)
    acc = accuracy_score(actuals, predictions)
    fu.log(workdir, f'[{i}] Done predict, acc={acc}', 
                   f'mem, ({fu.get_my_memory()})')
    
    y_prob_vec = fu.predict_proba(X_test_transformed, bundle_loc=bundle_loc)
    # xgb_model.predict_proba(X_test_transformed)
    fu.log(workdir, f'[{i}] Done fu.predict_proba', f'mem, ({fu.get_my_memory()})')
    
    
    logloss = fu.big_logloss(actuals, y_prob_vec, list(range(len(labels))))
    fu.log(workdir, f'[{i}] Done big_logloss, loss={logloss}.', 
                   f'mem, ({fu.get_my_memory()})')
                          
    # save full now though
    joblib.dump({'model': xgb_model,
                 'notebook': '2020-06-20-batch-batch.ipynb',
                'metrics': {'confusion': confusion,
                           'validation_logloss': logloss,
                           'validation_acc': acc},
                'dataset': {'v': 'v1', 'desc': 'neighborhood+gender'},
                'model': {'v': 'v1', 'desc': 'xgboost+defaults+onehot'}
                }, bundle_loc)
    fu.log(workdir, f'[{i}] dumped bundle to {bundle_loc}')
                             
```

    /opt/program/artifacts/2020-06-21T051742Z


#### truncated log

```
(pandars3) $ tail -f artifacts/2020-06-21T051742Z/work.log 
2020-06-21 05:17:42Z, Starting, , mem, ({'pmem': '34.7', 'rss': '0.678 GiB'})
2020-06-21 05:17:53Z, (0/43) Done fit, mem, ({'pmem': '36.0', 'rss': '0.704 GiB'})
2020-06-21 05:18:12Z, (1/43) Done fit, mem, ({'pmem': '36.0', 'rss': '0.704 GiB'})
2020-06-21 05:18:45Z, (2/43) Done fit, mem, ({'pmem': '36.3', 'rss': '0.709 GiB'})
2020-06-21 05:19:39Z, (3/43) Done fit, mem, ({'pmem': '36.7', 'rss': '0.717 GiB'})
2020-06-21 05:21:04Z, (4/43) Done fit, mem, ({'pmem': '37.1', 'rss': '0.725 GiB'})
2020-06-21 05:23:02Z, (5/43) Done fit, mem, ({'pmem': '37.5', 'rss': '0.733 GiB'})
2020-06-21 05:25:38Z, (6/43) Done fit, mem, ({'pmem': '37.9', 'rss': '0.741 GiB'})
2020-06-21 05:28:45Z, (7/43) Done fit, mem, ({'pmem': '38.3', 'rss': '0.749 GiB'})
2020-06-21 05:32:24Z, (8/43) Done fit, mem, ({'pmem': '38.8', 'rss': '0.758 GiB'})
2020-06-21 05:36:41Z, (9/43) Done fit, mem, ({'pmem': '39.3', 'rss': '0.768 GiB'})
2020-06-21 05:41:28Z, (10/43) Done fit, mem, ({'pmem': '39.8', 'rss': '0.777 GiB'})
2020-06-21 05:46:47Z, (11/43) Done fit, mem, ({'pmem': '40.2', 'rss': '0.787 GiB'})
2020-06-21 05:52:35Z, (12/43) Done fit, mem, ({'pmem': '40.7', 'rss': '0.796 GiB'})
2020-06-21 05:59:02Z, (13/43) Done fit, mem, ({'pmem': '41.3', 'rss': '0.807 GiB'})
...
...
2020-06-21 15:24:59Z, (41/43) Done fit, mem, ({'pmem': '55.5', 'rss': '1.085 GiB'})
2020-06-21 15:29:14Z, (42/43) Done fit, mem, ({'pmem': '56.0', 'rss': '1.094 GiB'})
2020-06-21 15:29:23Z, [0] Done fit., mem, ({'pmem': '56.0', 'rss': '1.094 GiB'})
```

#### Trying to understand what happened here..
- This notebook was meant to be a simple re-do of the [2020-06-20](https://github.com/namoopsoo/learn-citibike/blob/2020-revisit/notes/2020-06-20.md) notebook, except instead of running a `fit()` on all of the training data, `430k` rows, at once, to use the batching technique, `10k` at a time, 
- But this batching technique in the `do_train` func is not at all what happened. The first "2020-06-20" model fit took about `7 min` , but this fit here with `43` batches took from `05:17` to `11:23` , then `14:01` to `15:29` , maybe about `8 hours` . 
- The first "2020-06-20" model  per [here](https://github.com/namoopsoo/learn-citibike/blob/2020-revisit/notes/2020-06-20.md#log-dump) was about `2.9M` but this one, `105M` 
- This `43*2.9 = 124.7` so perhaps these `43` iterations are scaling the size roughly linearly.
- But this time around I was tracking the memory of my notebook (looks fine) and I made a habit of killing any previous notebooks. So I feel confident the slowness is related to whatever is happening in this step..

```
model.fit(X[part], y[part], xgb_model=prev_model_loc)
```

- If the predict parts took `5min` in "2020-06-20" then here (I'm still waiting for it as I write this `55minutes` in ) I may need to wait several hours .

#### What is next
- As long as I don't do that kind of batching, with at worst under an hour training cycle, I can run through a bunch of experiments, but as I am adding more and more data, I am sure I will run into more problems, so eventually I need to solve this batching issue.
- I'm wondering hmm maybe it is better to just train a new model for each batch, to make an ensemble of all the models in the batch. That's better than this slow process . [this answer](https://datascience.stackexchange.com/a/47537) seems to hint that yea, the additional `xgb_model` parameter to the `fit` method is not equivalent to just having taken in all the data at once. 
- Or also, perhaps the native `xgb.train` approach 
- [this answer](https://stackoverflow.com/a/44922590)  around pickling/unpickling seems to even say incremental learning with the sklearn API is not possible. So indeed I feel like I want to try it with the functional API.

#### num class self reminder
- [this tip](https://stackoverflow.com/questions/39386966/multiclass-classification-in-xgboost-python#39387627) about `num_class` was helpful but this implicitness means the first data which hits the classifier needs to have all the classes 
- oh right or just need to handle missing right.



#### Initial false start log..

```
model.xg  work.log  
(pandars3) $ tail -f artifacts/2020-06-21T051408Z/work.log 
2020-06-21 05:14:08Z, Starting, , mem, ({'pmem': '30.9', 'rss': '0.603 GiB'})
2020-06-21 05:14:12Z, (0/422) Done fit, mem, ({'pmem': '32.7', 'rss': '0.639 GiB'})
2020-06-21 05:14:15Z, (1/422) Done fit, mem, ({'pmem': '32.7', 'rss': '0.64 GiB'})
2020-06-21 05:14:19Z, (2/422) Done fit, mem, ({'pmem': '32.7', 'rss': '0.64 GiB'})
2020-06-21 05:14:27Z, (3/422) Done fit, mem, ({'pmem': '32.7', 'rss': '0.64 GiB'})
2020-06-21 05:14:38Z, (4/422) Done fit, mem, ({'pmem': '32.7', 'rss': '0.64 GiB'})
2020-06-21 05:14:52Z, (5/422) Done fit, mem, ({'pmem': '32.7', 'rss': '0.64 GiB'})
2020-06-21 05:15:09Z, (6/422) Done fit, mem, ({'pmem': '32.7', 'rss': '0.64 GiB'})
2020-06-21 05:15:29Z, (7/422) Done fit, mem, ({'pmem': '33.1', 'rss': '0.646 GiB'})
2020-06-21 05:15:52Z, (8/422) Done fit, mem, ({'pmem': '33.5', 'rss': '0.654 GiB'})
2020-06-21 05:16:18Z, (9/422) Done fit, mem, ({'pmem': '33.9', 'rss': '0.662 GiB'})
2020-06-21 05:16:47Z, (10/422) Done fit, mem, ({'pmem': '34.3', 'rss': '0.67 GiB'})
```

#### fuller log...

```
(pandars3) $ tail -f artifacts/2020-06-21T051742Z/work.log 
2020-06-21 05:17:42Z, Starting, , mem, ({'pmem': '34.7', 'rss': '0.678 GiB'})
2020-06-21 05:17:53Z, (0/43) Done fit, mem, ({'pmem': '36.0', 'rss': '0.704 GiB'})
2020-06-21 05:18:12Z, (1/43) Done fit, mem, ({'pmem': '36.0', 'rss': '0.704 GiB'})
2020-06-21 05:18:45Z, (2/43) Done fit, mem, ({'pmem': '36.3', 'rss': '0.709 GiB'})
2020-06-21 05:19:39Z, (3/43) Done fit, mem, ({'pmem': '36.7', 'rss': '0.717 GiB'})
2020-06-21 05:21:04Z, (4/43) Done fit, mem, ({'pmem': '37.1', 'rss': '0.725 GiB'})
2020-06-21 05:23:02Z, (5/43) Done fit, mem, ({'pmem': '37.5', 'rss': '0.733 GiB'})
2020-06-21 05:25:38Z, (6/43) Done fit, mem, ({'pmem': '37.9', 'rss': '0.741 GiB'})
2020-06-21 05:28:45Z, (7/43) Done fit, mem, ({'pmem': '38.3', 'rss': '0.749 GiB'})
2020-06-21 05:32:24Z, (8/43) Done fit, mem, ({'pmem': '38.8', 'rss': '0.758 GiB'})
2020-06-21 05:36:41Z, (9/43) Done fit, mem, ({'pmem': '39.3', 'rss': '0.768 GiB'})
2020-06-21 05:41:28Z, (10/43) Done fit, mem, ({'pmem': '39.8', 'rss': '0.777 GiB'})
2020-06-21 05:46:47Z, (11/43) Done fit, mem, ({'pmem': '40.2', 'rss': '0.787 GiB'})
2020-06-21 05:52:35Z, (12/43) Done fit, mem, ({'pmem': '40.7', 'rss': '0.796 GiB'})
2020-06-21 05:59:02Z, (13/43) Done fit, mem, ({'pmem': '41.3', 'rss': '0.807 GiB'})
2020-06-21 06:05:56Z, (14/43) Done fit, mem, ({'pmem': '41.8', 'rss': '0.817 GiB'})
2020-06-21 06:13:32Z, (15/43) Done fit, mem, ({'pmem': '42.3', 'rss': '0.826 GiB'})
2020-06-21 06:21:35Z, (16/43) Done fit, mem, ({'pmem': '42.8', 'rss': '0.836 GiB'})
2020-06-21 06:30:25Z, (17/43) Done fit, mem, ({'pmem': '43.3', 'rss': '0.846 GiB'})
2020-06-21 06:39:32Z, (18/43) Done fit, mem, ({'pmem': '43.8', 'rss': '0.856 GiB'})
2020-06-21 06:49:29Z, (19/43) Done fit, mem, ({'pmem': '44.3', 'rss': '0.866 GiB'})
2020-06-21 06:59:44Z, (20/43) Done fit, mem, ({'pmem': '44.8', 'rss': '0.876 GiB'})
2020-06-21 07:10:44Z, (21/43) Done fit, mem, ({'pmem': '45.4', 'rss': '0.887 GiB'})
2020-06-21 07:22:05Z, (22/43) Done fit, mem, ({'pmem': '45.8', 'rss': '0.896 GiB'})
2020-06-21 07:34:11Z, (23/43) Done fit, mem, ({'pmem': '46.3', 'rss': '0.905 GiB'})
2020-06-21 07:46:42Z, (24/43) Done fit, mem, ({'pmem': '46.8', 'rss': '0.915 GiB'})
2020-06-21 07:59:52Z, (25/43) Done fit, mem, ({'pmem': '47.5', 'rss': '0.929 GiB'})
2020-06-21 08:13:32Z, (26/43) Done fit, mem, ({'pmem': '48.0', 'rss': '0.937 GiB'})
2020-06-21 08:27:49Z, (27/43) Done fit, mem, ({'pmem': '48.4', 'rss': '0.946 GiB'})
2020-06-21 08:42:41Z, (28/43) Done fit, mem, ({'pmem': '48.9', 'rss': '0.955 GiB'})
2020-06-21 08:58:04Z, (29/43) Done fit, mem, ({'pmem': '49.4', 'rss': '0.965 GiB'})
2020-06-21 09:14:08Z, (30/43) Done fit, mem, ({'pmem': '49.9', 'rss': '0.975 GiB'})
2020-06-21 09:30:35Z, (31/43) Done fit, mem, ({'pmem': '50.4', 'rss': '0.985 GiB'})
2020-06-21 09:47:50Z, (32/43) Done fit, mem, ({'pmem': '50.9', 'rss': '0.994 GiB'})
2020-06-21 10:05:25Z, (33/43) Done fit, mem, ({'pmem': '51.5', 'rss': '1.006 GiB'})
2020-06-21 10:24:07Z, (34/43) Done fit, mem, ({'pmem': '52.2', 'rss': '1.02 GiB'})
2020-06-21 10:43:11Z, (35/43) Done fit, mem, ({'pmem': '52.7', 'rss': '1.029 GiB'})
2020-06-21 11:03:10Z, (36/43) Done fit, mem, ({'pmem': '53.1', 'rss': '1.038 GiB'})
2020-06-21 11:23:00Z, (37/43) Done fit, mem, ({'pmem': '53.6', 'rss': '1.048 GiB'})
2020-06-21 14:19:14Z, (38/43) Done fit, mem, ({'pmem': '54.1', 'rss': '1.057 GiB'})
2020-06-21 14:40:22Z, (39/43) Done fit, mem, ({'pmem': '54.6', 'rss': '1.066 GiB'})
2020-06-21 15:02:32Z, (40/43) Done fit, mem, ({'pmem': '55.1', 'rss': '1.076 GiB'})
2020-06-21 15:24:59Z, (41/43) Done fit, mem, ({'pmem': '55.5', 'rss': '1.085 GiB'})
2020-06-21 15:29:14Z, (42/43) Done fit, mem, ({'pmem': '56.0', 'rss': '1.094 GiB'})
2020-06-21 15:29:23Z, [0] Done fit., mem, ({'pmem': '56.0', 'rss': '1.094 GiB'})

```

#### size of first bundle on disk
```
(pandars3) $ ls -alrth artifacts/2020-06-21T051742Z
total 459144
drwxr-xr-x@ 41 michal  staff   1.3K Jun 21 01:17 ..
-rw-r--r--@  1 michal  staff   105M Jun 21 11:29 model.xg
-rw-r--r--@  1 michal  staff   3.7K Jun 21 11:29 work.log
drwxr-xr-x@  5 michal  staff   160B Jun 21 11:29 .
-rw-r--r--@  1 michal  staff   105M Jun 21 11:29 bundle_0.joblib
```
