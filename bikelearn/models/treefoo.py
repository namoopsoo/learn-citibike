import numpy as np
import pandas as pd
import cPickle

from sklearn.ensemble import RandomForestClassifier

from bikelearn import classify
from bikelearn import pipeline_data as pl
import bikelearn.settings as s


def make_tree_foo(trainset, stations):
    assert trainset['fn']
    assert stations['fn']

    cols = [s.NEW_START_POSTAL_CODE,
             s.NEW_START_BOROUGH, s.NEW_START_NEIGHBORHOOD,
             s.START_DAY, s.START_HOUR,
             s.AGE_COL_NAME, s.GENDER,
             s.USER_TYPE_COL]
    feature_encoding_dict = {
            s.USER_TYPE_COL: str,
            s.AGE_COL_NAME: float,
            s.NEW_START_POSTAL_CODE: str,
            s.NEW_START_BOROUGH: str, s.NEW_START_NEIGHBORHOOD:str,
            s.NEW_END_NEIGHBORHOOD:str}

    simpledf, label_encoders = pl.make_simple_df_from_raw(
            trainset['trainset'], stations['stations_df'],
            feature_encoding_dict)
    train_df, validation_df = classify.simple_split(simpledf) # FIXME: label encoders from full or train?

    X_train = np.array(train_df[cols])
    y_train = np.array(train_df[s.NEW_END_NEIGHBORHOOD])

    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train, y_train)

    # Quick simple evaluate on validation as well..
    X_validation = np.array(validation_df[cols])
    y_validation = np.array(validation_df[s.NEW_END_NEIGHBORHOOD])

    from nose.tools import set_trace; set_trace()

    y_predictions_validation = clf.predict(X_validation)
    zipped = zip(y_validation, y_predictions_validation)
    correct = len([[x,y] for x,y in zipped if x == y])
    proportion_correct = 1.0*correct/y_validation.shape[0]

    bundle = {
            'train_metadata': {'trainset_fn': trainset['fn'],
                'stations_df_fn': stations['fn'],
                'stations_df': stations['stations_df']},
            'timestamp': pl.make_timestamp(),
            'clf': clf,
            'model_id': 'tree-foo',
            'bundle_name': 'tree-foo-bundle',
            'label_encoders': label_encoders,
            'features': {
                'input': cols,
                'output_label': s.NEW_END_NEIGHBORHOOD,
                'dtypes': feature_encoding_dict},
            'evaluation': {'validation_proportion_correct':
                proportion_correct},
            'clf_info': {
                'feature_importances':
                zip(cols, clf.feature_importances_)}
            }
    return bundle


def set_model(pickled_bundle):
    pass


def do_predict(unpickled_bundle):
    label_encoders = unpickled_bundle['label_encoders']


