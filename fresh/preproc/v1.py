from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# dataset v1: start neighborhod and gender

def preprocess(X, y, neighborhoods):
    # Initially assuming labeled=True
    labeled = True
    genders = [0, 1, 2]

    enc = OneHotEncoder(handle_unknown='error', 
                        categories=[neighborhoods, genders])
    enc.fit(X)
    X_transformed = enc.transform(X)
    
    le = LabelEncoder()
    le.fit(y)  # previously on neighborhoods
    
    y_enc = le.transform(y)    
    
    return X_transformed, enc, le, y_enc

