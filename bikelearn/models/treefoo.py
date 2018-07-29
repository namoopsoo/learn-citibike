import numpy as np
import pandas as pd
import cPickle

from sklearn.ensemble import RandomForestClassifier

from bikelearn import classify
from bikelearn import pipeline_data as pl


def make_tree_foo(trainset, station_df):
    cols = ['start_postal_code', 'start_sublocality', 'start_neighborhood', 'start_day', 'start_hour', 'age', 'gender']

    simpledf = pl.make_simple_df_from_raw(trainset, station_df)

    train_df, holdout_df = classify.simple_split(simpledf, )

    X_train = np.array(train_df[cols])
    y_train = np.array(train_df['end_neighborhood'])

    clf = RandomForestClassifier(max_depth=2, random_state=0)

    clf.fit(X_train, y_train)

    bundle = {
            'clf': clf,
            'model_id': 'tree-foo',
            'bundle_name': 'tree-foo-bundle.pkl',
            
            }
    return bundle

