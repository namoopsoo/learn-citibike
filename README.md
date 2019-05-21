### Citibike Project: Can your Destination be Predicted

#### Motivation
I think sometimes the most interesting projects live behind ideas that sound impractical or even crazy. That's why I thought
it would be fun to use the Citibike bike share trip data to try and predict a person's destination based on what we know.

_Roughly speaking trip data looks like this_

```
"tripduration","starttime","stoptime","start station id","start station name","start station latitude","start station longitude","end station id","end station name","end station latitude","end station longitude","bikeid","usertype","birth year","gender"
"171","10/1/2015 00:00:02","10/1/2015 00:02:54","388","W 26 St & 10 Ave","40.749717753","-74.002950346","494","W 26 St & 8 Ave","40.74734825","-73.99723551","24302","Subscriber","1973","1"
"593","10/1/2015 00:00:02","10/1/2015 00:09:55","518","E 39 St & 2 Ave","40.74780373","-73.9734419","438","St Marks Pl & 1 Ave","40.72779126","-73.98564945","19904","Subscriber","1990","1"
"233","10/1/2015 00:00:11","10/1/2015 00:04:05","447","8 Ave & W 52 St","40.76370739","-73.9851615","447","8 Ave & W 52 St","40.76370739","-73.9851615","17797","Subscriber","1984","1"
"250","10/1/2015 00:00:15","10/1/2015 00:04:25","336","Sullivan St & Washington Sq","40.73047747","-73.99906065","223","W 13 St & 7 Ave","40.73781509","-73.99994661","23966","Subscriber","1984","1"
"528","10/1/2015 00:00:17","10/1/2015 00:09:05","3107","Bedford Ave & Nassau Ave","40.72311651","-73.95212324","539","Metropolitan Ave & Bedford Ave","40.71534825","-73.96024116","16246","Customer","","0"
"440","10/1/2015 00:00:17","10/1/2015 00:07:37","3107","Bedford Ave & Nassau Ave","40.72311651","-73.95212324","539","Metropolitan Ave & Bedford Ave","40.71534825","-73.96024116","23698","Customer","","0"
```

The data if fairly clean and regular, so I thought this was a fun data set to sharpen my teeth on.

#### Quick Bird's Eye  of my Journey
* First started just [looking](#more-on-this-data) at this data.
* Just out of curiosity, as a first mini starter project I decided to look at the [relationship between rider age and speed](#speed-and-age)
* I realized pretty early that the bike station target was too small, so I started using the Google Geolocation API to get broader location data such
as _zip codes_ and _neighborhoods_ . [geolocation](#need-additional-location-data)
* I also thought on a high level that knowing whether you got on your bike at `4:05` in the afternoon versus `4:06` shouldn't 
influence my learning algorithm, so I added some more [transformations](#time-bucketing).
* I compared the prediction accuracy of the new geolocation data as a [first stab](#comparing-geolocation-granularities)
* I ran through a couple of modeling scenarios [here](#deeper-into-the-weeds) and finding some impoartant inconsistencies in how I was running my testing
* I ended up getting some good gains by [binarizing my inputs](#binarizing-the-inputs)
* I ended up using a [different evaluation metric](#changing-my-metric-one-more-time) again to get a different perspective on this problem.
* Maybe a year or more later I came back to this problem from the point of view of experimenting with [AWS approaches to training models](#sagemaker-approach) , including Docker and SageMaker.
* Thoughts for [future improvements](#future-improvements)

#### More on this data
* When I started looking at this data, there were 400+ stations for docking your citibike.
* There is age, and some of the riders were actually born in the 1800s, which is kind of cool.

```python
df = load_data('data/201509_10-citibike-tripdata.csv.annotated.100000.06112016T1814.csv')

In [6]: df['birth year'].describe()
Out[6]: 
count    83171.000000
mean      1977.149680
std         11.400096
min       1885.000000
25%       1969.000000
50%       1980.000000
75%       1986.000000
max       1999.000000
Name: birth year, dtype: float64
```

#### Speed and Age
Turns out that you need to know the miles per the longitude degree at a particular latitude on our planet. So for our particular location, 
at lat around `40.723` and using the earth's radius of about `3958 miles` , we have about `52.3 miles/longitude degree`
here in NYC. 
 
So from there, looking at some of the speed data just involved looking at the tripdata trip time and calculating the 
cartesian distance. 
<img src="https://github.com/namoopsoo/learn-citibike/blob/master/notes/assets/Screen%20Shot%202019-05-21%20at%2011.02.41%20AM.png"
width="435" height="307"  >
1758 × 1238

(More on the code [here](https://github.com/namoopsoo/learn-citibike/blob/master/bikelearn/utils.py#L86) )
(Also more detail on this analysis in the main [jupyter notebook](https://github.com/namoopsoo/learn-citibike/blob/master/project%20report.ipynb)) 

#### Need additional location data
* With the 400+ stations, trying to predict a multi-class problem of this sort with basic machine learning algorithms
would not be a way to get quick results to help keep the project going, so I decided to constrain the problem. I ended up
turning to the Google Geocoding API. Using this data because a side project in its own right, because parsing through 
the Google geolocation data can get pretty hairy! 

_The meat of the output can look like this, for the docking station "1st Avenue & E 15th St"_
```python
{
'raw_result': [{u'address_components': [{u'long_name': u'1st Avenue',
                u'short_name': u'1st Avenue',
                u'types': [u'route']},
               {u'long_name': u'Midtown',
                u'short_name': u'Midtown',
                u'types': [u'neighborhood',
                           u'political']},
               {u'long_name': u'Manhattan',
                u'short_name': u'Manhattan',
                u'types': [u'sublocality_level_1',
                           u'sublocality',
                           u'political']},
               {u'long_name': u'New York',
                u'short_name': u'New York',
                u'types': [u'locality',
                           u'political']},
               {u'long_name': u'New York County',
                u'short_name': u'New York County',
                u'types': [u'administrative_area_level_2',
                           u'political']},
               {u'long_name': u'New York',
                u'short_name': u'NY',
                u'types': [u'administrative_area_level_1',
                           u'political']},
               {u'long_name': u'United States',
                u'short_name': u'US',
                u'types': [u'country',
                           u'political']},
               {u'long_name': u'10003',
                u'short_name': u'10003',
                u'types': [u'postal_code']}],
               u'formatted_address': u'1st Avenue & E 15th St, New York, NY 10003, USA',
               }
```


* Some of the challenges here were that the outputs from the API were not always consistent. The above output shows that the  
`'neighborhood'` is `'Midtown'`, but because there were 400+ stations, I did not initially notice that sometimes the `neighborhood`
was missing or that the `zip code` was missing. That ended up throwing off my code a couple of times. 
* It turned out that the _street intersection_ was not an ideal clean data input to the API. For instance `E 3 St & 1 Ave, NY` was understood by the API as just a street or a `route` as it is called, ( [raw output](https://github.com/namoopsoo/learn-citibike/blob/master/notes/assets/E%203%20St%20&%201%20Ave,%20NY.md) ). Later on I ended up refactoring this to 
use the raw _latitude and longitude_ .
* However, I eventually noticed that often times the _docking stations_ were on the edge of neighborhoods. So I literally had 
edge cases! The Neighborhood would come back blank and I ended up having to fill in a lot of that data by hand anyhow!

<img src="https://github.com/namoopsoo/learn-citibike/blob/master/notes/assets/Screen%20Shot%202018-12-04%20at%2012.11.42%20PM.png"
width="457" height="328">
1830 × 1314

* Also the data calls were not free and I ended up building a small _caching layer_ with _redis_ . 
* The other reason I had done that was that I would often work out of cafes where the Wifi was spotty and I didn't want an internet
connection to hold me back.
*  I think in hind-sight, I could have avoided some of the automation here and just decoupled the data fetch so that I wouldn't have to
worry about that internet connection. 
* Of course every time I wanted to add additional data from Citibike, there would be new docking stations and I had to get back to making sure
my station location data was correct, so that bad data did not impact predictions.

#### Time bucketing 
In order to get better information from the source time, the source time was bucketted into 24 hour-buckets per day. That is since a ride starting at 1:04:23pm shouldn't be treated as being too different from a ride departing at 1:05:24pm . There is more value in intuitively clustering the rides.

#### Comparing Geolocation Granularities
There are about 463 stations found in the dataset, 28 neighborhoods, representing 49 postal codes and 3 out of 5 boroughs,

So using  the `(start time bucket, start station id, age, gender)` as the inputs and  with `RandomizedLogisticRegression` as a classifier ,
for about a months worth of trip data, I saw roughly the following comparison.

```python
{'end station id': OrderedDict([('training',
               OrderedDict([('accuracy_score', 0.041432771986099973),
                            ('f1_score', 0.015138704086611844),
                            ('recall_score', 0.041432771986099973),
                            ('precision_score', 0.016942125433308568)])),
              ('holdout',
               OrderedDict([('accuracy_score', 0.031533939070016032),
                            ('f1_score', 0.0093952628045424723),
                            ('recall_score', 0.031533939070016032),
                            ('precision_score', 0.0067157290264759353)]))]),
 'end_neighborhood': OrderedDict([('training',
               OrderedDict([('accuracy_score', 0.39047231270358307),
                            ('f1_score', 0.28885663229134789),
                            ('recall_score', 0.39047231270358307),
                            ('precision_score', 0.26445041603375502)])),
              ('holdout',
               OrderedDict([('accuracy_score', 0.39630836047774159),
                            ('f1_score', 0.2935527390151364),
                            ('recall_score', 0.39630836047774159),
                            ('precision_score', 0.26579390443173939)]))]),
 'end_postal_code': OrderedDict([('training',
               OrderedDict([('accuracy_score', 0.14129127122042506),
                            ('f1_score', 0.068340173428106887),
                            ('recall_score', 0.14129127122042506),
                            ('precision_score', 0.068168259430770747)])),
              ('holdout',
               OrderedDict([('accuracy_score', 0.13361838588989844),
                            ('f1_score', 0.064738931917963718),
                            ('recall_score', 0.13361838588989844),
                            ('precision_score', 0.067139580345228156)]))]),
 'end_sublocality': OrderedDict([('training',
               OrderedDict([('accuracy_score', 0.95354786589470852),
                            ('f1_score', 0.95209028150037733),
                            ('recall_score', 0.95354786589470852),
                            ('precision_score', 0.95217920885515972)])),
              ('holdout',
               OrderedDict([('accuracy_score', 0.9493807215939688),
                            ('f1_score', 0.94990282092170586),
                            ('recall_score', 0.9493807215939688),
                            ('precision_score', 0.95132373575480056)]))])}
```

* The rough accuracies for prediction, end station id (~3%), postal code (~13%), neighborhood (~40%), and borough (~95%), using the small dataset, shows the rough differences in what happens when you reduce the number of possible outputs.
* Overall this gave me the motivation to focus on the `neighborhood` as a target to try to improve upon.
* More in the [jupyter notebook](https://render.githubusercontent.com/view/ipynb?commit=b2beb2af23f4f803a059161aeeb1a8e628a1bd4b&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6e616d6f6f70736f6f2f6c6561726e2d6369746962696b652f623262656232616632336634663830336130353931363161656562316138653632386131626434622f70726f6a6563742532307265706f72742e6970796e62&nwo=namoopsoo%2Flearn-citibike&path=project+report.ipynb&repository_id=60489657&repository_type=Repository#A-basic-learning-strategy-is-used)

#### Deeper into the weeds
I compared the `SGDClassifier` with the `LogisticRegression` classifier (which I believe just uses Gradient Descent, while the `SGDClassifier`  classifier is also a Logistic Regression classifier, but it uses Stochastic Gradient Descent).

I also tried applying [Standard Scaling](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) to my input data after [reading](http://scikit-learn.org/stable/modules/sgd.html#tips-on-practical-use) that  `scikit learn` 's `SGDClassifier`   implementation is sensitive unless the input data has a `mean=0` and `variance=1` .  Indeed per the below this helped a little.

<img src="https://github.com/namoopsoo/learn-citibike/blob/master/notes/assets/Screen%20Shot%202019-05-21%20at%2012.12.54%20PM.png"
width="637" height="302">

I also applied a GridSearch around the alpha parameter to the SGDClassifier, but this did not help at least the way I tried it,

<img src="https://github.com/namoopsoo/learn-citibike/blob/master/notes/assets/Screen%20Shot%202019-05-21%20at%2012.30.52%20PM.png"
width="690" height="450">

I next started varying the training set size, given that a month worth of trip data had about a million rows, I went from `10k` to `1M`,

<img src="https://github.com/namoopsoo/learn-citibike/blob/master/notes/assets/Screen%20Shot%202019-05-21%20at%2012.39.07%20PM.png"
width="678" height="235">

But this didn't look great. I realized a problem I had was that I was not randomly sampling my input data. Since a month-size dataset is around 1.2 Million rows, then a 10,000 large set just ends up barely dipping into the first day. So choosing the dataset sizes has to be done, by random sampling.

After making the sampling randomized, the output below, feels like it has a better upward trend, but it is still not visible enough.

<img src="https://github.com/namoopsoo/learn-citibike/blob/master/notes/assets/Screen%20Shot%202019-05-21%20at%2012.45.13%20PM.png"
width="682" height="235">

I also realized I was being inconsistent in my assessment because I was not using a single holdout set to test. I was actually randomly generating a test set each time. That was really bad. So I created ten models on sizes 10,000 through 100,000 datasets, created from `09 and 10 2015`, and testing on a single holdout dataset, taken from `November 2015`. In this approach, the accuracy results are found using the same holdout set instead of using a differently derived test set each time.

<img src="https://github.com/namoopsoo/learn-citibike/blob/master/notes/assets/Screen%20Shot%202019-05-21%20at%2012.54.28%20PM.png"
width="673" height="232">
Although the results were still pretty flat, at least I can trust the consistency of my test method more now.


More in the jupyter [notebook](https://render.githubusercontent.com/view/ipynb?commit=b2beb2af23f4f803a059161aeeb1a8e628a1bd4b&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6e616d6f6f70736f6f2f6c6561726e2d6369746962696b652f623262656232616632336634663830336130353931363161656562316138653632386131626434622f70726f6a6563742532307265706f72742e6970796e62&nwo=namoopsoo%2Flearn-citibike&path=project+report.ipynb&repository_id=60489657&repository_type=Repository#Also-comparing-with-additional-classifiers)

#### Binarizing the inputs
* Another important modeling change to try was to do a better of job of preparing the input data to better expose the stratification across citibike trips across the sources. To do this, instead of using a source station column and source neighborhood columns, the source neighborhood column was binarized using the sklearn OneHotEncoder, to a column for each of the neighborhoods in the surface area of the city.
* The same experiment as earlier was conducted, comparing results across the default SGDClassifier and LogisticRegression classifiers and also across 100,000 to 1,000,000 size datasets used for a train/test split along with a 100,000 large holdout set.
* These were created from just the single 2015-09 dataset (201509-citibike-tripdata.csv).
* I found this to be very helpful. Here's a summary graphic, 

<img src="https://github.com/namoopsoo/learn-citibike/blob/master/notes/assets/Screen%20Shot%202019-05-21%20at%201.00.56%20PM.png"
width="445" height="310">


More details [in the notebook](https://render.githubusercontent.com/view/ipynb?commit=b2beb2af23f4f803a059161aeeb1a8e628a1bd4b&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6e616d6f6f70736f6f2f6c6561726e2d6369746962696b652f623262656232616632336634663830336130353931363161656562316138653632386131626434622f70726f6a6563742532307265706f72742e6970796e62&nwo=namoopsoo%2Flearn-citibike&path=project+report.ipynb&repository_id=60489657&repository_type=Repository#Binarizing-geolocation-start-data)

#### Changing my metric one more time
* Given that there are about `28` neighborhoods covered by Citibike (at least in the data end of `2015`), a high accuracy is difficult to achieve especially because there are many output classes to choose from.
* Another idea that was explored was to create a Rank K Accuracy, such that a prediction is correct when the correct class is found in the top highest K probabilities.

<img src="https://github.com/namoopsoo/learn-citibike/blob/master/notes/assets/Screen%20Shot%202019-05-21%20at%201.07.21%20PM.png"
width="415" height="67">

The overall reasoning I had here is two-fold. One, I think of the analogy of a search engine, where it is typically acceptable to show someone 
_five results_ as opposed to the so called _"I am feeling lucky"_ result. Of course not every machine learning use case will have the tolerance to take five results as opposed to five, but I think my particular problem of choice it might be fine. Or at least asking people would help to answer that question.

But I think the main reason I wanted to do this was to just better understand whether my classification approach was doing anything at all. So since, out of these `28` or so neighborhoods, if going from the `top 1` result to the `top 2` results, yields an additional `20 points` of accuracy, then I feel a little better about the result making sense.

<img src="https://github.com/namoopsoo/learn-citibike/blob/master/notes/assets/Screen%20Shot%202019-05-21%20at%201.07.47%20PM.png"
width="334" height="239">


More detail in the [notebook](https://render.githubusercontent.com/view/ipynb?commit=b2beb2af23f4f803a059161aeeb1a8e628a1bd4b&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6e616d6f6f70736f6f2f6c6561726e2d6369746962696b652f623262656232616632336634663830336130353931363161656562316138653632386131626434622f70726f6a6563742532307265706f72742e6970796e62&nwo=namoopsoo%2Flearn-citibike&path=project+report.ipynb&repository_id=60489657&repository_type=Repository#Redefining-the-accuracy-score)


#### SageMaker approach
I wanted to test drive AWS SageMaker , 
* To see if I could more easily train models without relying on my laptop
* And make my training environment more reproducible
* And the prospect of hyper parameter tuning jobs seemed pretty neat too
* And I wanated to see just how simple serving models would be
* And in general I wanted to do all of these things to see if I could end up using this at my job (which I did).

#### What ended up happening
* I managed to recreate my basic model training and evaluation setup using Docker and Sagemaker. 
* With SageMaker, Docker, your model is more compartmentalized and I found this was helpful when iterating model code and boiler plate code.
* I also ended up serving the model on an endpoint.
* This is still a sort of a back burner project I would like to come back to at some point, 
* but below I show one update I made to have a slightly better time debugging [changes my model iterations](#changes-to-model-iterations)
* And I go over [several more data roadblocks](#bad-data-strikes-again) I ran into.

#### Changes to Model Iterations
To make things slightly easier to understand, especially for debugging purposes, I now made models into json objects that are easier to display
```python

localfn = '/Users/michal/Downloads/2018-12-07-update-model/2018-12-07-update-
   ...: model/tree-foo-bundle-pensive-swirles.2018-12-04T210259ZUTC.pkl'

In [5]: with open(localfn) as fd: bundle = cPickle.load(fd)

In [6]: from bikelearn import classify as blc

In [12]: blc.label_decode(bundle['label_encoders']['end_neighborhood'], range(40))
Out[12]: 
array(['-1', 'Alphabet City', 'Battery Park City', 'Bedford-Stuyvesant',
       'Boerum Hill', 'Bowery', 'Broadway Triangle', 'Brooklyn Heights',
       'Brooklyn Navy Yard', 'Central Park', 'Chelsea', 'Chinatown',
       'Civic Center', 'Clinton Hill',
       'Columbia Street Waterfront District', 'Downtown Brooklyn',
       'Dumbo', 'East Village', 'Financial District', 'Flatiron District',
       'Fort Greene', 'Fulton Ferry District', 'Garment District',
       'Gramercy Park', 'Greenpoint', 'Greenwich Village',
       "Hell's Kitchen", 'Hudson Square', 'Hunters Point', 'Kips Bay',
       'Korea Town', 'Lenox Hill', 'Lincoln Square', 'Little Italy',
       'Long Island City', 'Lower East Side', 'Lower Manhattan',
       'Meatpacking District', 'Midtown', 'Midtown East'], dtype=object)

In [19]: len(bundle['label_encoders']['end_neighborhood'].classes_)
Out[19]: 65

In [42]: bu.print_bundle(bundle)
Out[42]: 
{'bundle_name': 'tree-foo-bundle-pensive-swirles',
 'clf': RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
             max_depth=5, max_features='auto', max_leaf_nodes=None,
             min_impurity_decrease=0.0, min_impurity_split=None,
             min_samples_leaf=1, min_samples_split=2,
             min_weight_fraction_leaf=0.0, n_estimators=20, n_jobs=1,
             oob_score=False, random_state=0, verbose=0, warm_start=False),
 'clf_info': {'feature_importances': [('start_postal_code',
    0.4118465753431595),
   ('start_sublocality', 0.2228924462201325),
   ('start_neighborhood', 0.28008403752725497),
   ('start_day', 0.006151427471547376),
   ('start_hour', 0.03971509075090292),
   ('age', 0.0180403831546044),
   ('gender', 0.008144851450140815),
   ('usertype', 0.013125188082257624)]},
 'evaluation': {'test_metrics': {'confusion_matrix': 64,
   'f1_scores': {'macro': 0.041605116043552354,
    'micro': 0.1599860211928701,
    'weighted': 0.06355337311627869},
   'rank_k_proba_scores': {1: 0.1599860211928701,
    2: 0.2536814721966162,
    3: 0.3240640528912417,
    4: 0.3873466833318772,
    5: 0.443358194451626,
    10: 0.629839760791229}},
  'validation_metrics': {'confusion_matrix': 64,
   'f1_scores': {'macro': 0.04327900735885162,
    'micro': 0.16284068269032595,
    'weighted': 0.06549596580599053},
   'rank_k_proba_scores': {1: 0.16284068269032595,
    2: 0.2563720053782964,
    3: 0.3247784397051751,
    4: 0.3873943724175835,
    5: 0.4436898254016547,
    10: 0.6304384096730821}}},
 'features': {'dtypes': {'age': float,
   'end_neighborhood': str,
   'start_neighborhood': str,
   'start_postal_code': str,
   'start_sublocality': str,
   'usertype': str},
  'input': ['start_postal_code',
   'start_sublocality',
   'start_neighborhood',
   'start_day',
   'start_hour',
   'age',
   'gender',
   'usertype'],
  'output_label': 'end_neighborhood'},
 'label_encoders': {'age': LabelEncoder(),
  'end_neighborhood': LabelEncoder(),
  'start_neighborhood': LabelEncoder(),
  'start_postal_code': LabelEncoder(),
  'start_sublocality': LabelEncoder(),
  'usertype': LabelEncoder()},
 'model_id': 'tree-foo',
 'test_metadata': {'testset_fn': '/opt/ml/input/data/testing/201602-citibike-tripdata.csv'},
 'timestamp': '2018-12-04T210259ZUTC',
 'train_metadata': {'hyperparameters': {u'max_depth': u'5',
   u'n_estimators': u'20'},
  'stations_df_fn': '/opt/ml/input/data/training/support/stations-2018-12-04-c.csv',
  'trainset_fn': '/opt/ml/input/data/training/201601-citibike-tripdata.csv'}}

```


* I like how the AWS SageMaker setup allows for custom Docker image based models, only requiring a particular `csv` format as a data input for predictions and also requiring the Docker image implement a `train` command for running _training_ jobs.
* I ended up adding a `setup.py` to my git repo to version what code a particular `Dockerfile` would use 
* I made many iterations in trying to get the model on SageMaker up and running, so I liked the experience overall. 

#### Bad data strikes again
* After quickly updating docking station data and retraining, I ended up with a model which was returning only one value as an output
* I took a deeper dive into my docking station data and found yet again that I had a lot of blank geolocation neighborhood and postal code data.
* This is the same problem I had to deal with in the past as well.
* I ended up finding while debugging, that the empty data was essentially pinning the majority of the training data as `neighborhood : 'nan'`, and so the predictions, per this confusion matrix, were basically all the same output. 
```python
ipdb> pp skm.confusion_matrix(y_validation, y_predictions, classes)
array([[    0,     0,     0,     0,     0,     0,     0,   174],
       [    0,     0,     0,     0,     0,     0,     0,   116],
       [    0,     0,     0,     0,     0,     0,     0,   130],
       [    0,     0,     0,     0,     0,     0,     0,   357],
       [    0,     0,     0,     0,     0,     0,     0,   364],
       [    0,     0,     0,     0,     0,     0,     0,   255],
       [    0,     0,     0,     0,     0,     0,     0,   862],
       [    0,     0,     0,     0,     0,     0,     0, 97977]])
ipdb> 
```
![image](https://user-images.githubusercontent.com/2048242/48679839-d5bc2600-eb62-11e8-8d99-b6bdbf82e6e0.png)

* Of course one data problem always leads to another data problem. This time around, when I started updating my docking station geolocation data, I found that my payment information may have changed and so I was getting the following 
```
ipdb> pp geocoding_result
{u'error_message': u'You have exceeded your daily request quota for this API. If you did not set a custom daily request quota, verify your project has an active billing account: http://g.co/dev/maps-no-account',
 u'results': [],
 u'status': u'OVER_QUERY_LIMIT'}
 ```
* And of course I also ended up stepping on my own foot as well. I found I had created an unfortinate [git commit](https://github.com/namoopsoo/learn-citibike/commit/438e425482db1c105ae6a22b8248696d0f91dfef) where I accidentally undid the url-encoding and intersections with `&` are I think treated as query string parameters , , which ended up being a reason why some of my data was coming back as just the geolocation for a single street (`route`) and not an actual intersection. 

#### Future Improvements
* I am hoping to come back to this and continue to iterate the approach.
* In particular, I would like to continue to explore model degradation over time. 
* And in discussing with a few colleagues, seasonality would also be a really good feature to consider. Time bucketing was explored to a limited extent, but the day of the week nor the month of the year was not explored. 
* There may also be many other datasets which can be joined with this one to bolster the information available, including information about the weather or perhaps other demographic attributes available. 
* A more thorough comparison of algorithms should also be considered.
