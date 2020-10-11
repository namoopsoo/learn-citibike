


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
dtest = xgb.DMatrix(f'{train_loc}?format=libsvm')
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
