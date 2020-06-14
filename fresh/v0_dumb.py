import pandas as pd
import numpy as np
from scipy.special import softmax
from sklearn.metrics import log_loss


def make_probs(xdf):
    xdf['prob'] = softmax(xdf['count'])
    xdf['prob2'] = xdf['prob'].map(lambda x:round(x, ndigits=4))
    return pd.DataFrame(dict(
        list(xdf[['end_neighborhood', 'prob2']].to_records(index=False))),
                       index=[0])


def dumb_fit(X, y):
    mdf = X_y_to_mydf(X, y)
    statsdf = mdf[['start_neighborhood', 
               'end_neighborhood']].groupby(by=['start_neighborhood', 
               'end_neighborhood']).size().reset_index().rename(columns={0: 'count'})
    hmmdf = statsdf.groupby(by='start_neighborhood').apply(make_probs).fillna(0)
    
    split_data = hmmdf.to_dict(orient='split')
    lookup_dict = {a: np.array(split_data['data'])[i] 
              for (i, a) in enumerate([x[0] for x in split_data['index']])}
    labels = split_data['columns']
    return hmmdf, lookup_dict, labels
    
def X_y_to_mydf(X, y):
    return pd.DataFrame(np.vstack((X[:, 0], y)).T, columns=['start_neighborhood',
                                                   'end_neighborhood'])

class SimpleFlassifiler():
    def __init__(self):
        return
    def fit(self, X, y):
        (self.lookup_df, self.lookup_dict, 
             self.labels) = dumb_fit(X, y)
        
    def score(self, X, y_true):
        y_preds = predict(self.lookup_dict, X)
        return log_loss(y_true, y_preds, labels=self.labels)

    def get_params(self, deep):
        return {}

    def predict(self, X):
        return predict(self.lookup_dict, X)
    
def predict(lookup_dict, X):
    # return np.concatenate([lookup_dict[x[0]] for x in X])
    array_size = len(list(lookup_dict.values())[0])
    return np.concatenate([np.reshape(lookup_dict[x[0]], (1, array_size)) 
           for x in X])

