from collections import Counter
import numpy as np


def kth_correct(x, y, num_classes):
    # Given multiclass y_proba and the correct y_label, which k is correct?
    return [a[1] for a in sorted(list(zip(x, range(num_classes))), reverse=True)].index(y)


def kth_area(y_test, y_prob,
            num_classes):
    '''
    The k-area metric: Take the probabilities from a multi-class model,
    and for each example, rank the probabilities and for each find
    the rank k the model gives to the correct class.
    Then create an array from 1 through num_classes, with the "cumulative accuracy"
    for each position.
    The sum of this array can range from 0.0 to (num_classes-1)/num_classes

    y_prob: probabilities from a multi-class model
    y_test: correct labels
    '''
    size = y_test.shape[0]

    correct_kth = [kth_correct(y_prob[i], int(y_test[i]), num_classes)
                    for i in range(y_test.shape[0])]

    base_dict = {k: 0 for k in range(num_classes)}    
    correct_kth_counts = {**base_dict, **dict(Counter(correct_kth))}
    print('correct_kth_counts ', correct_kth_counts)

    cumulative_correct_kth = (
        lambda k, counts, size: sum([counts[i]/size for i in range(k)]))
    
    # k-Accuracy after considering each of k=0, 1,... (size-1) predictions.
    topk = np.array([cumulative_correct_kth(k, 
                        correct_kth_counts, size)/num_classes
            for k in range(num_classes)])
    print('topk', topk)

    # area = (topk/num_classes).sum()
    karea = topk.sum()
    return correct_kth, topk, karea


