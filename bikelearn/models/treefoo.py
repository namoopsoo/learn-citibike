import numpy as np
import pandas as pd
import cPickle
import funkybob

from sklearn.ensemble import RandomForestClassifier

from bikelearn import classify as blc
from bikelearn import pipeline_data as pl
import bikelearn.settings as s

import bikelearn.metrics_utils as blmu


def make_tree_foo(datasets, stations, hyperparameters={}):
    assert datasets['train_fn']
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
            s.NEW_START_BOROUGH: str, s.NEW_START_NEIGHBORHOOD: str,
            s.NEW_END_NEIGHBORHOOD: str}

    simpledf, label_encoders = pl.make_simple_df_from_raw(
            datasets['trainset'], stations['stations_df'],
            feature_encoding_dict)
    train_df, validation_df = blc.simple_split(simpledf) # FIXME: label encoders from full or train?


    X_train = np.array(train_df[cols])
    y_train = np.array(train_df[s.NEW_END_NEIGHBORHOOD])
    
    clf = RandomForestClassifier(
            max_depth=hyperparameters.get('max_depth'),
            n_estimators=hyperparameters.get('n_estimators'),
            random_state=0)
    clf.fit(X_train, y_train)

    validation_output = blmu.do_validation(clf, validation_df, cols)

    bundle = {
            'train_metadata': {'trainset_fn': datasets['train_fn'],
                'stations_df_fn': stations['fn'],
                'hyperparameters': hyperparameters},
            'timestamp': pl.make_timestamp(),
            'clf': clf,
            'model_id': 'tree-foo',
            'bundle_name': 'tree-foo-bundle-' + get_random_handle(),
            'label_encoders': label_encoders,
            'features': {
                'input': cols,
                'output_label': s.NEW_END_NEIGHBORHOOD,
                'dtypes': feature_encoding_dict},
            'evaluation': {
                'validation_metrics': validation_output},
            'clf_info': {
                'feature_importances':
                zip(cols, clf.feature_importances_)}
            }

    # Test...
    holdout_original_df = datasets['testset']  # FIXME properly handling this? 
    holdout_df = \
            blc.widen_df_with_other_cols(holdout_original_df, s.ALL_COLUMNS)
    
    _, _, test_output = blc.run_model_predict(
            bundle, holdout_df, stations['stations_df'], labeled=True)
    # test_output = blmu.do_validation(clf, holdout_df, cols)

    # update bundle.
    bundle['evaluation'].update({'test_metrics': test_output})
    bundle.update({
        'test_metadata': {'testset_fn': datasets['test_fn'],}
        })

    return bundle

def get_random_handle():
    generator = funkybob.RandomNameGenerator(members=2, separator='-')
    it = iter(generator)
    return it.next()


def set_model(pickled_bundle):
    pass


def do_predict(unpickled_bundle):
    label_encoders = unpickled_bundle['label_encoders']


