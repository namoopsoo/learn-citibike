from collections import Counter
import numpy as np

def kth_area(y_test, y_prob,
            num_classes):
    size = y_test.shape[0]

    kth_correct = (
        lambda x, y: [a[1] for a in sorted(list(zip(x, range(num_classes))), reverse=True)].index(y))

    correct_kth = [kth_correct(y_prob[i], int(y_test[i]))
                    for i in range(y_test.shape[0])]
    
    correct_kth_proportions = dict(Counter(correct_kth))

    cumulative_correct_kth = (
        lambda k, props, size: sum([props[i]/size for i in range(k)]))

    topk = np.array([cumulative_correct_kth(k, correct_kth_proportions, size)
            for k in range(num_classes)])

    return (topk/num_classes).sum()

