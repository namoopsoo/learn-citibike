import numpy as np
import pandas as pd
import cPickle

from sklearn.ensemble import RandomForestClassifier

from bikelearn import classify
from bikelearn import pipeline_data as pl


def make_tree_foo(trainset, stations):
    assert trainset['fn']
    assert stations['fn']
    cols = ['start_postal_code', 'start_sublocality', 'start_neighborhood', 'start_day', 'start_hour', 'age', 'gender']

    simpledf, label_encoders = pl.make_simple_df_from_raw(
            trainset['trainset'], stations['stations_df'])
    train_df, holdout_df = classify.simple_split(simpledf)

    X_train = np.array(train_df[cols])
    y_train = np.array(train_df['end_neighborhood'])

    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train, y_train)

    # Quick simple evaluate on holdout as well..
    X_holdout = np.array(holdout_df[cols])
    y_holdout = np.array(holdout_df['end_neighborhood'])

    y_predictions_holdout = clf.predict(X_holdout)
    zipped = zip(y_holdout, y_predictions_holdout)
    correct = len([[x,y] for x,y in zipped if x == y])
    proportion_correct = correct/y_holdout.shape[0]*1.0

    bundle = {
            'train_metadata': {'trainset_fn': trainset['fn'],
                'stations_df_fn': stations['fn']},
            'timestamp': pl.make_timestamp(),
            'clf': clf,
            'model_id': 'tree-foo',
            'bundle_name': 'tree-foo-bundle',
            'label_encoders': label_encoders,
            'evaluation': {'holdout_proportion_correct':
                proportion_correct}
            }
    return bundle


def set_model(pickled_bundle):
    pass


def do_predict(unpickled_bundle):
    label_encoders = unpickled_bundle['label_encoders']


