

import numpy as np
from collections import OrderedDict 
from sklearn.pipeline import Pipeline
from sklearn.linear_model import (LogisticRegression,
    RandomizedLogisticRegression)
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import (auc, accuracy_score, f1_score, 
        roc_curve, recall_score, precision_score)

from sklearn.preprocessing import LabelEncoder



def prepare_datas(df, features=None, feature_encoding=None,
        label_col=None):
    '''
ipdb> pp classifier.fit(np.array(datas['X_train']), 
                    np.array(datas['y_train'][u'end station id']))

    '''


    if feature_encoding:
        label_encoders = {}

        for feature in feature_encoding:
            if feature not in df.columns:
                continue

            label_encoders[feature] = LabelEncoder()

            df[feature] = label_encoders[feature].fit_transform(
                            df[feature])

    if features:
        X = df[features]

    else:
        X = df[df.columns[:-1]]

    if label_col:
        y = df[label_col]
    else:
        y = df[df.columns[-1]]

    X_train, X_holdout, y_train, y_holdout = train_test_split(
            X, y, test_size=0.2)

    out = {
            # All data
            'X': np.array(X),
            'y': np.array(y),
            'X_train': np.array(X_train), 
            'X_holdout': np.array(X_holdout), 
            'y_train': np.array(y_train), 
            'y_holdout': np.array(y_holdout),
            }

    return out
    
def build_classifier():

    rlr_feature_selection = RandomizedLogisticRegression(
        C=1.5, n_jobs=-1, verbose=0)

    # Standard sklearn classifier
    lr_classifier = LogisticRegression(C=1.5)

    clf = Pipeline([
#        ('string_encoder', pp_encode_strings),
#        ('drop_nan_cols', pp_drop_nan_cols),
#        ('fix_collinear', pp_fix_collinear),
#
#        ('float_imputer', pp_imputer),
#        ('scaler', pp_scaler),
        # ('feature_selection', rlr_feature_selection),
        ('classification', lr_classifier )
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

