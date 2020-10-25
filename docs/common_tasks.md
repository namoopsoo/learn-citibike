

#### Loading data

```python
datadir = '/opt/data'
localdir = '/opt/program'
tripsdf = pd.read_csv(f'{datadir}/2013-07 - Citi Bike trip data.csv')
stationsdf = pd.read_csv(f'{localdir}/datas/stations/stations-2018-12-04-c.csv',
                        index_col=0)

```


#### Validating model predictions
* As [used here](https://github.com/namoopsoo/learn-citibike/blob/2020-revisit/notes/2020-07-16-local.md#i-would-like-to-get-the-train-acc-too-to-better-understand-the-overunder-aka-the-overfittingunderfitting)

```python
import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, accuracy_score, balanced_accuracy_score


artifacts_dir = 'some_artifacts_dir'
data_dir = 'my_local_dir'
test_loc = f'{datadir}/test.libsvm'
dtest = xgb.DMatrix(f'{test_loc}?format=libsvm')
actuals = dtest.get_label()

# If an xgb model is in a bundle like so..
bundle = joblib.load(f'{artifactsdir}/a_bundle.joblib')
model = bundle['xgb_model']

# Can get probabilities and class predictions
y_prob_vec = model.predict(dtrain)
predictions = np.argmax(y_prob_vec, axis=1)

logloss = log_loss(actuals, y_prob_vec, labels=list(range(54)))

acc = accuracy_score(actuals, predictions)
balanced_acc = balanced_accuracy_score(actuals, predictions)

```
K area metric too

```python
import fresh.metrics as fm
correct_kth, topk, karea = fm.kth_area(actuals, y_prob_vec, num_classes=54)
```

#### Mapping XGboost model features to names
* Also [this note](https://github.com/namoopsoo/learn-citibike/blob/master/notes/2020-07-26-feature-importances.md) has the basics around pulling together feature names and xgboost generic features (i.e. `f0, f1, ...`)
* Given a joblib bundle w/ an xgboost model,

```python
bundle_loc = f'bundle_with_metrics.joblib'
bundle = joblib.load(bundle_loc)
model = bundle['xgb_model']
```

The generic feature names are available with

```python
model.feature_names[:5]
# ['f0', 'f1', 'f2', 'f3', 'f4']
```
* But also a feature map is available through
```python
import fresh.preproc.v2 as pv2
feature_map = pv2.make_feature_map(bundle['proc_bundle']['bundle'])
# =>
{'f0': 'start_neighborhood=Alphabet City',
 'f1': 'start_neighborhood=Battery Park City',
 'f2': 'start_neighborhood=Bedford-Stuyvesant',
 'f3': 'start_neighborhood=Bloomingdale',
 'f4': 'start_neighborhood=Boerum Hill',
 'f5': 'start_neighborhood=Bowery',
 'f6': 'start_neighborhood=Broadway Triangle',
 'f7': 'start_neighborhood=Brooklyn Heights',
 'f8': 'start_neighborhood=Brooklyn Navy Yard',
 'f9': 'start_neighborhood=Carnegie Hill',
 'f10': 'start_neighborhood=Carroll Gardens',
 'f11': 'start_neighborhood=Central Park',
 'f12': 'start_neighborhood=Chelsea',
 'f13': 'start_neighborhood=Chinatown',
 'f14': 'start_neighborhood=Civic Center',
 'f15': 'start_neighborhood=Clinton Hill',
 'f16': 'start_neighborhood=Cobble Hill',
 'f17': 'start_neighborhood=Columbia Street Waterfront District',
 'f18': 'start_neighborhood=Downtown Brooklyn',
 'f19': 'start_neighborhood=Dumbo',
 'f20': 'start_neighborhood=East Harlem',
 'f21': 'start_neighborhood=East Village',
 'f22': 'start_neighborhood=East Williamsburg',
 'f23': 'start_neighborhood=Financial District',
 'f24': 'start_neighborhood=Flatiron District',
 'f25': 'start_neighborhood=Fort Greene',
 'f26': 'start_neighborhood=Fulton Ferry District',
 'f27': 'start_neighborhood=Garment District',
 'f28': 'start_neighborhood=Governors Island',
 'f29': 'start_neighborhood=Gowanus',
 'f30': 'start_neighborhood=Gramercy Park',
 'f31': 'start_neighborhood=Greenpoint',
 'f32': 'start_neighborhood=Greenwich Village',
 'f33': "start_neighborhood=Hell's Kitchen",
 'f34': 'start_neighborhood=Hudson Square',
 'f35': 'start_neighborhood=Hunters Point',
 'f36': 'start_neighborhood=Kips Bay',
 'f37': 'start_neighborhood=Korea Town',
 'f38': 'start_neighborhood=Lenox Hill',
 'f39': 'start_neighborhood=Lincoln Square',
 'f40': 'start_neighborhood=Little Italy',
 'f41': 'start_neighborhood=Long Island City',
 'f42': 'start_neighborhood=Lower East Side',
 'f43': 'start_neighborhood=Lower Manhattan',
 'f44': 'start_neighborhood=Meatpacking District',
 'f45': 'start_neighborhood=Midtown',
 'f46': 'start_neighborhood=Midtown East',
 'f47': 'start_neighborhood=Midtown West',
 'f48': 'start_neighborhood=Murray Hill',
 'f49': 'start_neighborhood=NoHo',
 'f50': 'start_neighborhood=NoMad',
 'f51': 'start_neighborhood=Nolita',
 'f52': 'start_neighborhood=Park Slope',
 'f53': 'start_neighborhood=Peter Cooper Village',
 'f54': 'start_neighborhood=Prospect Heights',
 'f55': 'start_neighborhood=Prospect Park',
 'f56': 'start_neighborhood=Red Hook',
 'f57': 'start_neighborhood=Rose Hill',
 'f58': 'start_neighborhood=SoHo',
 'f59': 'start_neighborhood=Stuyvesant Heights',
 'f60': 'start_neighborhood=Stuyvesant Town',
 'f61': 'start_neighborhood=Sunset Park',
 'f62': 'start_neighborhood=Sutton Place',
 'f63': 'start_neighborhood=Theater District',
 'f64': 'start_neighborhood=Tribeca',
 'f65': 'start_neighborhood=Tudor City',
 'f66': 'start_neighborhood=Two Bridges',
 'f67': 'start_neighborhood=Ukrainian Village',
 'f68': 'start_neighborhood=Union Square',
 'f69': 'start_neighborhood=Upper East Side',
 'f70': 'start_neighborhood=Upper West Side',
 'f71': 'start_neighborhood=Vinegar Hill',
 'f72': 'start_neighborhood=West Village',
 'f73': 'start_neighborhood=Williamsburg',
 'f74': 'start_neighborhood=Yorkville',
 'f75': 'gender=0',
 'f76': 'gender=1',
 'f77': 'gender=2',
 'f78': 'time_of_day=0',
 'f79': 'time_of_day=1',
 'f80': 'time_of_day=2',
 'f81': 'time_of_day=3',
 'f82': 'time_of_day=4',
 'f83': 'usertype',
 'f84': 'weekday'}
```

#### Extract top feature F Scores
* As per my notes [here](https://github.com/namoopsoo/learn-citibike/blob/master/notes/2020-10-21-look-at-model-plot.md)
```python
import pandas as pd
import fresh.preproc.v2 as pv2

feature_map = pv2.make_feature_map(bundle['proc_bundle']['bundle'])
df = pd.DataFrame.from_records([{'name': feature_map.get(k), 'f': k, 'fscore': v} for (k,v) in model.get_fscore().items()])

# Rank by Fscore
df.sort_values(by='fscore', ascending=False).iloc[:30]
```

#### Plotting an xgboost model's trees
* As per my notes [here](https://github.com/namoopsoo/learn-citibike/blob/master/notes/2020-10-21-look-at-model-plot.md)
* `fmap` is another parameter available to the below `plot_tree` but I have not tried it out yet.
```python
from xgboost import plot_tree
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(50, 50))
tree = 0
plot_tree(model, ax=ax, num_trees=tree)
# plt.show()
plt.savefig(
    f"/opt/downloads/model_id_{fpu.extract_model_id_from_bundle(bundle)}_tree{tree}.png")
```

#### From s3

```python
import fresh.s3utils as fs3
mybucket = os.getenv('MY_BUCKET')

s3uri = (f's3://{mybucket}/'
         'blah/blah/file.csv'
         )
bucket, s3fn = fs3.s3uri_to_parts(s3uri)
data = fs3.read_s3_file(bucket, s3fn)
```


#### Build/Run docker image...
* I have [docker notes here](/notes/2020-06-07-local-docker-notes.md) but I might move them here for convenience.


#### Deploy html
* Put my html to S3 static website bucket

```
make deploy_html
```

#### build a new revision of the lambda
```
make lambda
```
