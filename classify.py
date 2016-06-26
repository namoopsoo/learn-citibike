

import numpy as np
from collections import OrderedDict 
from sklearn.pipeline import Pipeline
from sklearn.linear_model import (LogisticRegression,
    RandomizedLogisticRegression, SGDClassifier)
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import (auc, accuracy_score, f1_score, 
        roc_curve, recall_score, precision_score)

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.grid_search import GridSearchCV

from pipeline_data import (remove_rows_with_nulls,
        re_index)


from utils import dump_np_array

def encode_holdout_df(holdout_df, label_encoders, feature_encoding):
    '''

    Replace any values which are not known, w/ a -1. 
    The label encoders have all mapped -1 or '-1' to 0, so this will avoid the
        ValueError which would otherwise come up when faced with values
        which were not previously seen by the label encoders.
    '''
    holdout_copy = holdout_df.copy()
    for feature in feature_encoding:
        if feature not in holdout_copy.columns:
            continue

        # what is the dtype
        if holdout_copy[feature].dtype == int:
            dtype = float
        elif holdout_copy[feature].dtype == np.object:
            dtype = str

        # Need to encode any unknown value as -1.
        # So first need to do a map for any values not known, map them to be -1.
        replacer = replace_unknown(label_encoders[feature].classes_, dtype)
        holdout_copy[feature] = holdout_copy[feature].apply(replacer)

        holdout_copy[feature] = label_encoders[feature].fit_transform(
                    holdout_copy[feature])

    return holdout_copy


def build_label_encoders_from_df(df, feature_encoding):
    #
    label_encoders = {}

    dfcopy = df.copy()

    for feature in feature_encoding:
        if feature not in dfcopy.columns:
            continue

        label_encoders[feature] = LabelEncoder()

        label_encoders[feature].fit(
                        np.concatenate([dfcopy[feature], np.array([-1])])
                        )
        dfcopy[feature] = label_encoders[feature].transform(
                        dfcopy[feature])

    return dfcopy, label_encoders



def replace_unknown(values, dtype):
    def replacer(v):
        if v not in values:
            if dtype == str:
                return '-1'
            elif dtype == float:
                return -1
        else:
            return v

    return replacer

def prepare_datas(df, holdout_df=None, features=None, feature_encoding=None,
        feature_standard_scaling=None,
        label_col=None):
    '''
ipdb> pp classifier.fit(np.array(datas['X_train']), 
                    np.array(datas['y_train'][u'end station id']))

preparing a holdout set:
    - reuse the label encoders from the training, but for the holdout set,
    need to check if values are not in the encoding set, then need to replace w/ -1 .
    '''
    # Remove nulls...
    df_unnulled = remove_rows_with_nulls(df)
    df_re_index = re_index(df_unnulled)
    df = df_re_index

    # Remove nulls from holdout too
    if holdout_df is not None:
        holdout_df_unnulled = remove_rows_with_nulls(holdout_df)
        holdout_df_re_index = re_index(holdout_df_unnulled)
        holdout_df = holdout_df_re_index

    if feature_encoding:

        label_encoders = {}
        dfcopy, label_encoders = build_label_encoders_from_df(df, feature_encoding)
        df = dfcopy

        if holdout_df is not None:
            holdout_df = encode_holdout_df(holdout_df, label_encoders, feature_encoding)


    if features:
        X = df[features]

    else:
        X = df[df.columns[:-1]]

    if label_col:
        y = df[label_col]
    else:
        y = df[df.columns[-1]]

    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)


    if feature_standard_scaling:
        # Train 
        scaler = StandardScaler().fit(X_train)
        X_train_out  = scaler.transform(X_train)
        X_test_out = scaler.transform(X_test)

        # TODO: same scaler or different scaler on whole X?
        #   probably on whole X, and then that scaler needs to be
        #   stored away for transforming new input vectors which come in to be predicted.
        X_full = StandardScaler().fit_transform(X)
    else:
        X_train_out = X_train
        X_test_out = X_test
        X_full = X
       
    dump_np_array(X_full, 'dump')

    # TODO: how to also return the label encoders and scalers

    out = {
            # All data
            'X': np.array(X_full),
            'y': np.array(y),
            'X_train': np.array(X_train_out), 
            'X_test': np.array(X_test_out), 
            'y_train': np.array(y_train), 
            'y_test': np.array(y_test),
            }

    return out

def grid_search_params(data):
    ''' Find best params for given data, for SGDClassifier
    '''

    classifier = SGDClassifier(n_iter=10**2)

    parameters = {'alpha': 10.0**-np.arange(1,7)}

    clf = GridSearchCV(classifier, parameters)
    clf.fit(data['X'], data['y'])

    best_params = clf.best_params_
    return best_params

    
def build_classifier(definition, datas):

    if definition['classification'] == 'lr':
        classifier = LogisticRegression(C=1.5)
    elif definition['classification'] == 'sgd':
        classifier = SGDClassifier(alpha=0.0001, n_iter = 10**2)
    elif definition['classification'] == 'sgd_grid':

        best_params = grid_search_params(datas)
        classifier = SGDClassifier(n_iter = 10**2, **best_params)


    rlr_feature_selection = RandomizedLogisticRegression(
        C=1.5, n_jobs=-1, verbose=0)

    # Standard sklearn classifier

    clf = Pipeline([
#        ('string_encoder', pp_encode_strings),
#        ('drop_nan_cols', pp_drop_nan_cols),
#        ('fix_collinear', pp_fix_collinear),
#
#        ('float_imputer', pp_imputer),
#        ('scaler', pp_scaler),
        # ('feature_selection', rlr_feature_selection),
        ('classification', classifier )
    ])

    return clf

def run_predictions(classifier, X):
    y_predictions = classifier.predict(X)
    return y_predictions

def run_metrics_on_predictions(y_true, y_predictions):

    my_metrics = [accuracy_score, f1_score, 
        roc_curve, recall_score, precision_score]

    results = OrderedDict()

    for metrik in my_metrics:
        try:
            result = metrik(y_true, y_predictions)

            results[metrik.__name__] = result
        except ValueError as e:
            print 'couldnt run metrik ', metrik, ' => ', e

    return results

