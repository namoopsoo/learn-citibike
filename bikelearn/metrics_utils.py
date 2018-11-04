import numpy as np
import bikelearn.settings as s


def do_validation(clf, validation_df, cols):

    X_validation = np.array(validation_df[cols])
    y_validation = np.array(validation_df[s.NEW_END_NEIGHBORHOOD])

    y_predictions = clf.predict(X_validation)
    classes = clf.classes_
    y_predict_proba = clf.predict_proba(X_validation)

    metrics = gather_metrics(y_validation, y_predictions, y_predict_proba, classes)
    return metrics


def gather_metrics(y_test, y_predictions, y_predict_proba, classes):

    metrics = {
        'rank_k1_proba_score': rank_k_proba_score(y_test, y_predict_proba, classes, k=1),
        'rank_k2_proba_score': rank_k_proba_score(y_test, y_predict_proba, classes, k=2),
        'rank_k3_proba_score': rank_k_proba_score(y_test, y_predict_proba, classes, k=3),
        'rank_k4_proba_score': rank_k_proba_score(y_test, y_predict_proba, classes, k=4),
        'rank_k5_proba_score': rank_k_proba_score(y_test, y_predict_proba, classes, k=5),
        'rank_k10_proba_score': rank_k_proba_score(y_test, y_predict_proba, classes, k=10),
        # 'proportion_correct_foo': get_proportion_correct(y_test, y_predictions)
        }
    return metrics


def rank_k_proba_score(y_test, y_predict_proba, classes_, k=None):
    y_topk_outputs = get_sorted_predict_proba_predictions(y_predict_proba, classes_, k)
    return get_proportion_correct(y_test, y_topk_outputs)


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
