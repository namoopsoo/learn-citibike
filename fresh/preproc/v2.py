import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# dataset v2: start neighborhod and gender
#   and weeday|weekend
#   and customer|subscriber
#   and birth year one hot..

def preprocess(X, y, neighborhoods, labeled, proc_dict=None):
    num_rows = X.shape[0]

    if not labeled:
        X_transformed = xform(X, prod_dict)
    else:

        #
        'usertype'  # 'Customer', 'Subscriber'
        genders = [0, 1, 2]
        user_types = ['Subscriber', 'Customer']
        time_of_day = [0, 1, 2, 3, 4]
        features = ['start_neighborhood', 'gender', 'time_of_day', 'usertype', 'weekday', ]

        enc = OneHotEncoder(handle_unknown='error', 
                            categories=[neighborhoods, genders, time_of_day])

        # first 3 ....
        enc.fit(X[:, :3])
        X_transformed = enc.transform(X[:, :3])

        usertype_le = LabelEncoder()
        usertype_le.fit(X[:, 3])
        
        le = LabelEncoder()
        le.fit(y)  # previously on neighborhoods

        proc_dict = {'enc': enc, 'usertype_le': usertype_le, 'le': le}
        
        y_enc = le.transform(y)    
        
        return X_transformed, enc, le, y_enc

def xform(X, prod_dict):
    X_transformed = np.hstack((
        proc_dict['enc'].transform(X[:, :3]),
        np.resize(
            proc_dict['usertype_le'].transform(X[:, 3]),
            (num_rows, 1)
            ),
        X[:, 4:5]
        ))
    return X_transformed

