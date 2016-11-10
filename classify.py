

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
        re_index, feature_binarization)


from utils import dump_np_array

def what_is_dtype(s):
    if s.dtype == int:
        dtype = float
    elif s.dtype == float:
        dtype = float
    elif s.dtype == np.object:
        dtype = str
    else:
        raise Exception, 'what is the dtype' + str(s.dtype)

    return dtype

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

        dtype = what_is_dtype(holdout_copy[feature])

        # Need to encode any unknown value as -1.
        # So first need to do a map for any values not known, map them to be -1.
        replacer = replace_unknown(label_encoders[feature].classes_, dtype)
        holdout_copy[feature] = holdout_copy[feature].apply(replacer)

        holdout_copy[feature] = label_encoders[feature].transform(
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

        dtype = what_is_dtype(dfcopy[feature])
        if dtype == int:
            missing_val = -1 
        elif dtype == str:
            missing_val = '-1'
        else:
            raise Exception, 'unknown dtype' + str(dtype)

        # The missing value is fit as the last value in the label encoder. 
        #   So then when a new value is encountered afterwards, it is changed
        #   into the missing value.
        label_encoders[feature].fit(
                        np.concatenate([dfcopy[feature], np.array([missing_val])])
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
        label_col=None, one_hot_encoding=None):
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

        # Get the encoed dataframe.
        dfcopy, label_encoders = build_label_encoders_from_df(df, feature_encoding)
        df = dfcopy

        if holdout_df is not None:
            holdout_df = encode_holdout_df(holdout_df, label_encoders, feature_encoding)


    if features:
        X = df[features]

        if holdout_df is not None:
            X_holdout = holdout_df[features]
    else:
        X = df[df.columns[:-1]]

        if holdout_df is not None:
            X_holdout = holdout_df[holdout_df.columns[:-1]]

    # One hot here.
    if one_hot_encoding:
        # For each input feature being encoded, we want the new columns concatenated
        #   into the output.
        X = feature_binarization(X, one_hot_encoding)

        if X_holdout is not None:
            X_holdout = feature_binarization(X_holdout, one_hot_encoding)

    # FIXME ... use the appropriate LabelEncoder here. Unless already done by the t_t_split()
    if label_col:
        y = df[label_col]

        if holdout_df is not None:
            y_holdout = holdout_df[label_col]
    else:
        y = df[df.columns[-1]]

        if holdout_df is not None:
            y_holdout = holdout_df[holdout_df.columns[-1]]

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
        scaler_full = StandardScaler().fit(X)
        X_full = scaler_full.transform(X)

        # 
        if holdout_df is not None:
            X_holdout_scaled = scaler_full.transform(X_holdout)
    else:
        X_train_out = X_train
        X_test_out = X_test
        X_full = X

        if holdout_df is not None:
            X_holdout_scaled = X_holdout
       
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

    if holdout_df is not None:
        out.update({
            'X_holdout': np.array(X_holdout_scaled),
            'y_holdout': np.array(y_holdout)
            })

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

