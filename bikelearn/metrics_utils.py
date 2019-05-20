import numpy as np

import sklearn.metrics as skm
import bikelearn.settings as s


def do_validation(clf, validation_df, cols):

    X_validation = np.array(validation_df[cols])
    y_validation = np.array(validation_df[s.NEW_END_NEIGHBORHOOD])

    y_predictions = clf.predict(X_validation)
    classes = clf.classes_
    y_predict_proba = clf.predict_proba(X_validation)

    metrics = gather_metrics(y_validation, y_predictions, y_predict_proba, classes)
    return metrics


def gather_metrics(y_true, y_predictions, y_predict_proba, classes):

    metrics = {
            'rank_k_proba_scores': 
            {k: rank_k_proba_score(y_true, y_predict_proba, classes, k=k)
                for k in [1, 2, 3, 4, 5, 10]},
            'f1_scores': 
            {average_func:
                skm.f1_score(y_true, y_predictions, labels=classes,
                    average=average_func)
                for average_func in ['micro', 'macro', 'weighted']},
            'confusion_matrix': get_confusion_matrix(
                y_true, y_predictions, classes)
            }
    return metrics


def rank_k_proba_score(y_true, y_predict_proba, classes_, k=None):
    y_topk_outputs = get_sorted_predict_proba_predictions(y_predict_proba, classes_, k)
    return get_proportion_correct(y_true, y_topk_outputs)


def get_confusion_matrix(y_true, y_pred, classes):
    return skm.confusion_matrix(y_true, y_pred, labels=classes).tolist()


def get_proportion_correct(y_validation, y_predictions_validation):
    zipped = zip(y_validation, y_predictions_validation)
    correct = len([[x,y] for x,y in zipped if x in y and y != 'nan'])
    proportion_correct = 1.0*correct/y_validation.shape[0]
    return proportion_correct

def get_sorted_predict_proba_predictions(out_probabilities, classes, k=None):

#     first_row = out_probabilities[0]
#     first_grouped = zip(first_row, classes)
#     first_sorted_list = sorted(first_grouped, key=lambda x:x[0])
#     first_sorted_predictions = [x[1] for x in first_sorted_predictions]

    v1 = [zip(row, classes) for row in out_probabilities]
    v2 = [sorted(row, key=lambda x:x[0], reverse=True) for row in v1]
    v3 = [[x[1] for x in row]
            for row in v2]

    if k is None:
        return v3

    v4 = [x[:k] for x in v3]
    return v4
