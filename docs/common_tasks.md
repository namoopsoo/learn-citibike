


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
* I have [docker notes here](notes/2020-06-07-local-docker-notes.md) but I might move them here for convenience.


#### Deploy html
* Put my html to S3 static website bucket

```
make deploy_html
```

#### build a new revision of the lambda
```
make lambda
```
