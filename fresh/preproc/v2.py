import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import fresh.utils as fu

# dataset v2: start neighborhod and gender
#   and weeday|weekend
#   and customer|subscriber
#   and birth year one hot..

def preprocess(X, y, neighborhoods, labeled, proc_bundle=None, workdir=None):
    num_rows = X.shape[0]

    if proc_bundle:
        X_transformed, y_enc = xform(proc_bundle, X, y)
        return X_transformed, y_enc
    else:
        'usertype'  # 'Customer', 'Subscriber'
        genders = [0, 1, 2]
        user_types = ['Subscriber', 'Customer']
        time_of_day = [0, 1, 2, 3, 4]
        features = ['start_neighborhood', 'gender', 'time_of_day', 'usertype', 'weekday', ]

        enc = OneHotEncoder(handle_unknown='error', 
                            categories=[neighborhoods, genders, time_of_day])
        # first 3 .... NOTE: make this ordering less error prone
        enc.fit(X[:, :3])

        usertype_le = LabelEncoder()
        usertype_le.fit(X[:, 3])
        
        le = LabelEncoder()
        le.fit(y)  # previously had used neighborhoods here
        proc_bundle = {'enc': enc, 'usertype_le': usertype_le, 'le': le}
        X_transformed, y_enc = xform(proc_bundle, X, y, workdir)
        
        return X_transformed, y_enc, proc_bundle
        

def xform(proc_bundle, X, y=None, workdir):
    '''Apply preprocessing to X, y.  '''
    # TODO ... also the y_enc part, 
    # , for which need to also handle missing..
    #
    num_rows = X.shape[0]
    slices = fu.get_slices(list(range(num_rows)), num_slices=10)

    X_transformed_parts = []
    for a, b in tqdm(slices):
        X_transformed = np.hstack((
            proc_bundle['enc'].transform(X[a:b, :3]).toarray(),
            np.resize(
                proc_bundle['usertype_le'].transform(X[a:b, 3]),
                ((b - a), 1)
                ),
            X[a:b, 4:5]
            ))
        X_transformed_parts.append(X_transformed)
    X_transformed = np.concatenate(X_transformed_parts)

    if y is not None:
        y_enc = proc_bundle['le'].transform(y)    
    else:
        y_enc = None
    return X_transformed, y_enc


#proc_bundle['enc'].transform(X[:, :3]), np.resize(proc_bundle['usertype_le'].transform(X[:, 3]), (num_rows, 1)), X[:, 4:5]
# *** ValueError: all the input arrays must have same number of dimensions, but the array at index 0 has 1 dimension(s) and the array at index 1 has 2 dimension(s)
# ((843416, 83), (843416, 1), (843416, 1)) 

#p np.hstack((np.resize(proc_bundle['usertype_le'].transform(X[:, 3]), (num_rows, 1)), proc_bundle['enc'].transform(X[:, :3]), X[:, 4:5]))
# *** ValueError: all the input arrays must have same number of dimensions, but the array at index 0 has 2 dimension(s) and the array at index 1 has 1 dimension(s)
