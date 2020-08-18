

* [here](https://github.com/namoopsoo/learn-citibike/blob/2020-revisit/notes/2020-07-26-feature-importances.md) I also have code around the proc bundle and associating feature names to features helpfully, which I did not have earlier

```python
datadir = f'/opt/program/artifacts/2020-07-08T143732Z'
proc_bundle = joblib.load(f'{datadir}/proc_bundle.joblib')

'artifacts/2020-07-08T143732Z/proc_bundle.joblib'
```
* and I also refer to my list of models from July's tuning,

```python
artifactsdir = 'artifacts/2020-07-10T135910Z'
alldf = pd.read_csv(f'{artifactsdir}/train+test_results_vec.csv')
alldf.iloc[0]

Out[30]:
train_acc                   0.00797076
train_balanced_acc           0.0185185
i                                  868
train_logloss                  34.2635
train_karea                   0.473045
max_depth                            4
learning_rate                        1
objective               multi:softprob
num_class                           54
base_score                         0.5
booster                         gbtree
colsample_bylevel                  0.1
colsample_bynode                     1
colsample_bytree                   0.1
gamma                                0
max_delta_step                       0
min_child_weight                     1
random_state                         0
reg_alpha                            0
reg_lambda                           1
scale_pos_weight                     1
seed                                42
subsample                          0.4
verbosity                            0
acc                         0.00806245
balanced_acc                 0.0185185
logloss                        34.2602
walltime                       718.513
karea                         0.472606
num_round                           80
train_test_acc_delta       9.16906e-05
Name: 0, dtype: object


In [36]: alldf.sort_values(by='karea', ascending=False)[['i', 'karea', 'acc', 'train_acc', 'train_balanced_acc', 'num_round']].iloc[:5]                                                                                            
Out[36]:
         i     karea       acc  train_acc  train_balanced_acc  num_round
1253  1187  0.760827  0.121715   0.126969            0.110183        100
1252  1241  0.760578  0.122530   0.128123            0.111239        100
1251  1169  0.759658  0.122160   0.126160            0.108063        100
1250  1181  0.759652  0.121838   0.124693            0.106591        100
1249  1223  0.759477  0.122303   0.126950            0.109864        100



```
* prediction,
```python
i = 1187
bundle = joblib.load(f'{artifactsdir}/{i}_bundle_with_metrics.joblib')
model = bundle['xgb_model']

y_prob_vec = model.predict(dtrain)
predictions = np.argmax(y_prob_vec, axis=1)


```

#### Ok so what is missing?
* What the top model per above, is `i=1187`
* make an end to end predict script using these bundles
* and make a bundle that includes both preproc plus model bundle
