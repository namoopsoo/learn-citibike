import numpy as np
try:
    from tqdm import tqdm
except:
    pass
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.datasets import dump_svmlight_file
import fresh.utils as fu


def preprocess(X, y=None, neighborhoods=None, proc_bundle=None, workdir=None, dataset_name=None):
    '''
    Expecting:
features: ['start_neighborhood', 'gender', 'time_of_day', 'usertype', 'weekday', 'age_bin']

X: array(['Union Square',  1, 3, 'Subscriber', 1, 2.0])

    '''
    num_rows = X.shape[0]
    if proc_bundle:
        if y is not None:
            outfile = xform(proc_bundle, X, y, workdir, dataset_name, filetype='libsvm')
            return outfile
        else:
            return xform(proc_bundle, X)
    else:
        genders = [0, 1, 2]
        user_types = ['Subscriber', 'Customer']
        time_of_day = [0, 1, 2, 3, 4]
        features = ['start_neighborhood', 'gender', 'time_of_day', 'usertype', 'weekday', 
                    'age_bin'] # TODO

        enc = OneHotEncoder(handle_unknown='error', 
                            categories=[neighborhoods, genders, time_of_day])
        # first 3 .... NOTE: make this ordering less error prone
        enc.fit(X[:, :3])

        usertype_le = LabelEncoder()
        usertype_le.fit(X[:, 3])
        
        le = LabelEncoder()
        le.fit(y)  # previously had used neighborhoods here
        proc_bundle = {'enc': enc, 'usertype_le': usertype_le, 'le': le}
        outfile = xform(proc_bundle, X, y, workdir, dataset_name, filetype='libsvm')
        return proc_bundle, outfile
        

def xform(proc_bundle, X, y=None, workdir=None, dataset_name=None, filetype=None):
    '''Apply preprocessing to X, y.  '''
    num_rows = X.shape[0]
    slices = fu.get_slices(list(range(num_rows)), num_slices=1) # NOTE hmm I think this is a bug if more than 1 slice 
    for a, b in tqdm(slices):
        slice_size = b - a
        stacks = [
            proc_bundle['enc'].transform(X[a:b, :3]).toarray(),
            np.resize(
                proc_bundle['usertype_le'].transform(X[a:b, 3]),
                (slice_size, 1)
                ),
            X[a:b, 4:5],

            # TODO last column..
            # X[a:b, 5:6]
            np.nan_to_num(X[a:b, 5:6].astype(np.float64), nan=-1)
            ]
        X_transformed = np.hstack(stacks)

        if y is not None:
            y_enc = proc_bundle['le'].transform(y[a:b])
        else:
            y_enc = None

        outfile = f'{workdir}/{dataset_name or "data"}.{filetype}'
        if filetype == 'csv':
            to_csv(X_transformed, y_enc, outfile)
        elif filetype == 'libsvm':
            to_libsvm(X_transformed, y_enc, outfile)
        elif filetype is None:
            # FIXME returning in for loop
            return X_transformed
    return outfile


def to_csv(X, y, outfile):
    stacks = [X]
    if y is not None:
        stacks.insert(0, np.resize(
            y,
            (y.shape, 1)))
    data = np.hstack(stacks)
    with open(outfile, 'ab') as fd:
        np.savetxt(fd, data, delimiter=',', fmt='%u')


def to_libsvm(X, y, outfile):
    with open(outfile, 'ab') as fd:
        dump_svmlight_file(X, y, fd)


# def make_feature_map(proc_bundle): # TODO
