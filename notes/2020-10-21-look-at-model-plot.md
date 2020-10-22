
### Summary
- Here I took a look at xgboost feature mapping again as a refresher, 
- Also getting feature fscores
- as well as plotting individual trees with graphviz
- And I experienced some difficulty after building a new Docker image w/ graphviz, because the scikit learn version got updated and loading a bundle stopped working normally. But going back to the original docker image with the original scikitlearn version restored bundle loading and prediction. 


```python
import matplotlib.pyplot as plt
```


```python
from importlib import reload
import pandas as pd
```


```python
import fresh.preproc.v2 as pv2
import fresh.predict_utils as fpu
bundle = fpu.load_bundle_in_docker()
```

    Loading from bundle_loc /opt/ml/model/all_bundle_with_stationsdf.joblib



```python
from xgboost import plot_tree
```


```python
model = bundle['model_bundle']['bundle']['xgb_model']
```

#### Looking for a quick model id candidate


```python
# bundle['model_bundle']['original_filename']#.keys()
# '/opt/program/artifacts/2020-07-10T135910Z/1187_bundle_with_metrics.joblib'
parts = bundle['model_bundle']['original_filename'].split('/')
model_id = f"{parts[4]}_{parts[5].split('_')[0]}"
model_id
```




    '2020-07-10T135910Z_1187'



#### plotting a tree from xgboost with graphviz


```python
# fig = plt.figure(figsize=(50,50))
# ax = fig.add_subplot(111)
fig, ax = plt.subplots(figsize=(50, 50))
tree = 0
plot_tree(model, ax=ax, num_trees=tree)
# plt.show()
plt.savefig(
    f"/opt/downloads/model_id_{fpu.extract_model_id_from_bundle(bundle)}_tree{tree}.png")
```


![png](2020-10-21-look-at-model-plot_files/2020-10-21-look-at-model-plot_9_0.png)



```python
plt.savefig(f"/opt/downloads/model_id_{fpu.extract_model_id_from_bundle(bundle)}_tree{tree}.png")
```


    <Figure size 432x288 with 0 Axes>



```python
fpu.extract_model_id_from_bundle(bundle)
```




    '2020-07-10T135910Z_1187'



#### feature map


```python
print(model.feature_names[:5])

```

    ['f0', 'f1', 'f2', 'f3', 'f4']



```python
feature_map = pv2.make_feature_map(bundle['proc_bundle']['bundle'])
feature_map
```




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



#### Rank features by Fscore


```python

# Rank by Fscore
model.get_fscore()
df = pd.DataFrame.from_records([{'name': feature_map.get(k), 'f': k, 'fscore': v} for (k,v) in model.get_fscore().items()])
```


```python
df.sort_values(by='fscore', ascending=False).iloc[:30]
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
      <th>name</th>
      <th>f</th>
      <th>fscore</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>weekday</td>
      <td>f84</td>
      <td>12812</td>
    </tr>
    <tr>
      <th>10</th>
      <td>gender=1</td>
      <td>f76</td>
      <td>8973</td>
    </tr>
    <tr>
      <th>2</th>
      <td>time_of_day=3</td>
      <td>f81</td>
      <td>8377</td>
    </tr>
    <tr>
      <th>8</th>
      <td>gender=0</td>
      <td>f75</td>
      <td>7969</td>
    </tr>
    <tr>
      <th>11</th>
      <td>time_of_day=1</td>
      <td>f79</td>
      <td>7064</td>
    </tr>
    <tr>
      <th>26</th>
      <td>time_of_day=2</td>
      <td>f80</td>
      <td>6594</td>
    </tr>
    <tr>
      <th>7</th>
      <td>time_of_day=0</td>
      <td>f78</td>
      <td>6302</td>
    </tr>
    <tr>
      <th>17</th>
      <td>gender=2</td>
      <td>f77</td>
      <td>5509</td>
    </tr>
    <tr>
      <th>3</th>
      <td>time_of_day=4</td>
      <td>f82</td>
      <td>4854</td>
    </tr>
    <tr>
      <th>40</th>
      <td>start_neighborhood=Chelsea</td>
      <td>f12</td>
      <td>1199</td>
    </tr>
    <tr>
      <th>37</th>
      <td>start_neighborhood=Midtown East</td>
      <td>f46</td>
      <td>1058</td>
    </tr>
    <tr>
      <th>36</th>
      <td>start_neighborhood=Midtown West</td>
      <td>f47</td>
      <td>947</td>
    </tr>
    <tr>
      <th>30</th>
      <td>start_neighborhood=Downtown Brooklyn</td>
      <td>f18</td>
      <td>910</td>
    </tr>
    <tr>
      <th>41</th>
      <td>start_neighborhood=Hell's Kitchen</td>
      <td>f33</td>
      <td>877</td>
    </tr>
    <tr>
      <th>21</th>
      <td>start_neighborhood=Fort Greene</td>
      <td>f25</td>
      <td>865</td>
    </tr>
    <tr>
      <th>14</th>
      <td>start_neighborhood=Financial District</td>
      <td>f23</td>
      <td>860</td>
    </tr>
    <tr>
      <th>23</th>
      <td>start_neighborhood=Brooklyn Heights</td>
      <td>f7</td>
      <td>834</td>
    </tr>
    <tr>
      <th>49</th>
      <td>start_neighborhood=Kips Bay</td>
      <td>f36</td>
      <td>821</td>
    </tr>
    <tr>
      <th>13</th>
      <td>start_neighborhood=Tribeca</td>
      <td>f64</td>
      <td>813</td>
    </tr>
    <tr>
      <th>28</th>
      <td>start_neighborhood=Lower East Side</td>
      <td>f42</td>
      <td>786</td>
    </tr>
    <tr>
      <th>38</th>
      <td>start_neighborhood=Theater District</td>
      <td>f63</td>
      <td>745</td>
    </tr>
    <tr>
      <th>39</th>
      <td>start_neighborhood=Midtown</td>
      <td>f45</td>
      <td>736</td>
    </tr>
    <tr>
      <th>5</th>
      <td>start_neighborhood=Greenwich Village</td>
      <td>f32</td>
      <td>733</td>
    </tr>
    <tr>
      <th>19</th>
      <td>start_neighborhood=Clinton Hill</td>
      <td>f15</td>
      <td>703</td>
    </tr>
    <tr>
      <th>33</th>
      <td>start_neighborhood=Chinatown</td>
      <td>f13</td>
      <td>695</td>
    </tr>
    <tr>
      <th>20</th>
      <td>start_neighborhood=Williamsburg</td>
      <td>f73</td>
      <td>683</td>
    </tr>
    <tr>
      <th>48</th>
      <td>start_neighborhood=Murray Hill</td>
      <td>f48</td>
      <td>681</td>
    </tr>
    <tr>
      <th>31</th>
      <td>start_neighborhood=Dumbo</td>
      <td>f19</td>
      <td>680</td>
    </tr>
    <tr>
      <th>44</th>
      <td>start_neighborhood=Civic Center</td>
      <td>f14</td>
      <td>660</td>
    </tr>
    <tr>
      <th>12</th>
      <td>start_neighborhood=Battery Park City</td>
      <td>f1</td>
      <td>649</td>
    </tr>
  </tbody>
</table>
</div>



#### Access to predict methods of individual trees


```python
dump_list = model.get_dump()
num_t=len(dump_list)
```


```python
num_t
```




    5400




```python
import numpy as np; from numpy import float32, array

```


```python
blahvec = [array([-0.19117647], dtype=float32), array([-0.17493302], dtype=float32), array([-0.1632322], dtype=float32), array([-0.15528071], dtype=float32), array([-0.14764059], dtype=float32), array([-0.134592], dtype=float32), array([-0.10049492], dtype=float32), array([-0.1302135], dtype=float32), array([-0.1268928], dtype=float32), array([-0.11612129], dtype=float32), array([-0.08275878], dtype=float32), array([-0.1172291], dtype=float32), array([-0.11398101], dtype=float32), array([-0.1115346], dtype=float32), array([-0.11064661], dtype=float32), array([-0.07345426], dtype=float32), array([-0.10717297], dtype=float32), array([-0.10513449], dtype=float32), array([-0.10398841], dtype=float32), array([-0.10136986], dtype=float32), array([-0.06175613], dtype=float32), array([-0.0963707], dtype=float32), array([-0.07870603], dtype=float32), array([-0.09415579], dtype=float32), array([-0.09520769], dtype=float32), array([-0.09691024], dtype=float32), array([-0.08932781], dtype=float32), array([0.04739833], dtype=float32), array([-0.09100223], dtype=float32), array([-0.07952356], dtype=float32), array([-0.05331874], dtype=float32), array([-0.08192921], dtype=float32), array([-0.07854772], dtype=float32), array([-0.08773494], dtype=float32), array([0.04309559], dtype=float32), array([-0.08494949], dtype=float32), array([-0.0410471], dtype=float32), array([-0.08940816], dtype=float32), array([-0.09339404], dtype=float32), array([-0.01659751], dtype=float32), array([-0.06599617], dtype=float32), array([0.02654409], dtype=float32), array([-0.08834934], dtype=float32), array([0.02391124], dtype=float32), array([-0.08487129], dtype=float32), array([-0.08129168], dtype=float32), array([-0.04954338], dtype=float32), array([-0.07444715], dtype=float32), array([-0.04855251], dtype=float32), array([-0.07878494], dtype=float32), array([-0.04307795], dtype=float32), array([-0.07436275], dtype=float32), array([-0.05560017], dtype=float32), array([-0.01397228], dtype=float32), array([-0.07300758], dtype=float32), array([-0.06667089], dtype=float32), array([-0.00923872], dtype=float32), array([-0.04245901], dtype=float32), array([-0.06061077], dtype=float32), array([-0.01046324], dtype=float32), array([-0.03374958], dtype=float32), array([-0.05404854], dtype=float32), array([-0.06557703], dtype=float32), array([-0.05135775], dtype=float32), array([0.03413153], dtype=float32), array([-0.03066111], dtype=float32), array([-0.05308104], dtype=float32), array([-0.0292182], dtype=float32), array([0.03178072], dtype=float32), array([-0.05284214], dtype=float32), array([-0.00391817], dtype=float32), array([-0.03907442], dtype=float32), array([-0.05048847], dtype=float32), array([-0.04551744], dtype=float32), array([-0.04168606], dtype=float32), array([-0.02567053], dtype=float32), array([-0.01324797], dtype=float32), array([-0.05392694], dtype=float32), array([-0.04365301], dtype=float32), array([-0.03963995], dtype=float32), array([-0.04544497], dtype=float32), array([-0.01181078], dtype=float32), array([-0.04806614], dtype=float32), array([-0.03913403], dtype=float32), array([0.00593662], dtype=float32), array([-0.00613165], dtype=float32), array([-0.03568554], dtype=float32), array([-0.03988504], dtype=float32), array([-0.03771305], dtype=float32), array([-0.00819349], dtype=float32), array([-0.04160738], dtype=float32), array([-0.03248644], dtype=float32), array([-0.00781107], dtype=float32), array([-0.02349615], dtype=float32), array([-0.03252411], dtype=float32), array([-0.03434181], dtype=float32), array([-0.01068497], dtype=float32), array([0.00462055], dtype=float32), array([-0.02988434], dtype=float32), array([-0.03110838], dtype=float32)]
```


```python
len(blahvec)

```




    100




```python
! conda install --yes scikit-learn=0.22.1 
# ! conda install --help
```

    Collecting package metadata (current_repodata.json): done
    Solving environment: failed with initial frozen solve. Retrying with flexible solve.
    Collecting package metadata (repodata.json): done
    Solving environment: done
    
    ## Package Plan ##
    
      environment location: /opt/conda
    
      added / updated specs:
        - scikit-learn=0.22.1
    
    
    The following packages will be downloaded:
    
        package                    |            build
        ---------------------------|-----------------
        scikit-learn-0.22.1        |   py37hd81dba3_0         5.2 MB
        ------------------------------------------------------------
                                               Total:         5.2 MB
    
    The following packages will be DOWNGRADED:
    
      scikit-learn                        0.23.2-py37h0573a6f_0 --> 0.22.1-py37hd81dba3_0
    
    
    
    Downloading and Extracting Packages
    scikit-learn-0.22.1  | 5.2 MB    | ##################################### | 100% 
    Preparing transaction: done
    Verifying transaction: done
    Executing transaction: done



```python
'''
# oops needed to downgrade this
/opt/conda/lib/python3.7/site-packages/sklearn/base.py:334: UserWarning: Trying to unpickle estimator OneHotEncoder from version 0.22.1 when using version 0.23.2. This might lead to breaking code or invalid results. Use at your own risk.
  UserWarning)
/opt/conda/lib/python3.7/site-packages/sklearn/base.py:334: UserWarning: Trying to unpickle estimator LabelEncoder from version 0.22.1 when using version 0.23.2. This might lead to breaking code or invalid results. Use at your own risk.
  UserWarning)
'''
bundle = fpu.load_bundle_in_docker()
```

    Loading from bundle_loc /opt/ml/model/all_bundle_with_stationsdf.joblib


    /opt/conda/lib/python3.7/site-packages/sklearn/base.py:334: UserWarning: Trying to unpickle estimator OneHotEncoder from version 0.22.1 when using version 0.23.2. This might lead to breaking code or invalid results. Use at your own risk.
      UserWarning)
    /opt/conda/lib/python3.7/site-packages/sklearn/base.py:334: UserWarning: Trying to unpickle estimator LabelEncoder from version 0.22.1 when using version 0.23.2. This might lead to breaking code or invalid results. Use at your own risk.
      UserWarning)



```python
# bundle = fpu.load_bundle_in_docker()
import sklearn
print(sklearn.__version__)
from importlib import reload
reload(sklearn)
print(sklearn.__version__)
```

    0.23.2
    0.22.1



```python
bundle = fpu.load_bundle_in_docker()
```

    Loading from bundle_loc /opt/ml/model/all_bundle_with_stationsdf.joblib



```python
reload(fpu)
```




    <module 'fresh.predict_utils' from '/opt/program/fresh/predict_utils.py'>




```python


record = {
     'starttime': '2013-07-01 00:00:00',
     'start station name': 'E 47 St & 2 Ave',
     'usertype': 'Customer',
     'birth year': '1999',
     'gender': 0
     }
# X_test = xgb.DMatrix()
a, b = fpu.full_predict_v2(bundle, record)
```

      0%|          | 0/1 [00:00<?, ?it/s]

    ['start_neighborhood', 'gender', 'time_of_day', 'usertype', 'weekday']
    [['Midtown East' 0 4 'Customer' 1]]
    [[0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
      0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
      0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
      0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
      0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0 1]]
    [02:51:44] 1x85 matrix with 4 entries loaded from /opt/server/hmmmm.libsvm?format=libsvm


    



```python
a, b
```




    (array([[0.01845511, 0.0112169 , 0.00608919, 0.00358758, 0.02280447,
             0.00894809, 0.00416442, 0.01859882, 0.0227448 , 0.01681302,
             0.0115949 , 0.00805871, 0.00163983, 0.01053979, 0.00618187,
             0.01423946, 0.01279947, 0.01256619, 0.01246874, 0.00309547,
             0.02578371, 0.02672892, 0.01732285, 0.03873935, 0.00959205,
             0.04965306, 0.01366295, 0.02828944, 0.00437104, 0.02245426,
             0.00961021, 0.00906411, 0.03759988, 0.13236406, 0.04010247,
             0.05108095, 0.00328353, 0.0088856 , 0.01417809, 0.00381548,
             0.00600353, 0.01121624, 0.00846388, 0.02701668, 0.01104608,
             0.04947915, 0.01212523, 0.01051854, 0.00749311, 0.03515568,
             0.01406138, 0.00103403, 0.01931338, 0.01388424]], dtype=float32),
     array([33]))




```python
a.sum()
```




    0.99999994




```python

```
