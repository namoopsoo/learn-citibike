

#### Use entropy to measure uncertainty
* Like [here](https://towardsdatascience.com/entropy-is-a-measure-of-uncertainty-e2c000301c2c)

#### xgboost access to individual trees is a feature request hmm
* Per [stackoverflow](https://datascience.stackexchange.com/questions/57905/how-to-extract-trees-in-xgboost) , [this](https://github.com/dmlc/xgboost/issues/2175) and [this](https://github.com/dmlc/xgboost/issues/3439) and xgboost feature issues around producing predictions for every tree
* Those issues are not yet resolved, but [this github link](https://github.com/bmreiniger/datascience.stackexchange/blob/master/57905.ipynb) has a working

#### what about change up random state to measure uncertainty?
```python
xg_clas=xgb.XGBClassifier(random_state=1, n_estimators=100)
```
