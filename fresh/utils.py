import datetime
import math
import pytz
import os
import numpy as np
from sklearn.metrics import log_loss


def utc_ts():
    return datetime.datetime.utcnow(
        ).replace(tzinfo=pytz.UTC).strftime('%Y-%m-%dT%H%M%SZ')

def utc_log_ts():
    return datetime.datetime.utcnow(
        ).replace(tzinfo=pytz.UTC).strftime('%Y-%m-%d %H:%M:%SZ')
    
def make_work_dir():
    ts = utc_ts()
    workdir = f'/opt/program/artifacts/{ts}' 
    os.mkdir(workdir)
    return workdir

def prepare_data(tripsdf, stationsdf):
    
    # Step 1, merge w/ stationsdf to get neighborhood data
    mdf = tripsdf[['start station name', 'end station name', 'gender']
            ].merge(stationsdf[['station_name', 'neighborhood']], 
                    left_on='start station name',
                    right_on='station_name'
                   ).rename(columns={'neighborhood': 'start_neighborhood'}
                           ).merge(stationsdf[['station_name', 'neighborhood']],
                                  left_on='end station name',
                                   right_on='station_name'
                                  ).rename(columns={'neighborhood': 'end_neighborhood'})
    
    neighborhoods = sorted(stationsdf.neighborhood.unique().tolist())
    
    X, y = (mdf[['start_neighborhood', 'gender']].values, 
            np.array(mdf['end_neighborhood'].tolist()))
    return X, y, neighborhoods
    

def get_partitions(vec, slice_size, keep_remainder=True):
    try:
        size = len(vec)
    except:
        size = vec.shape[0]

    assert slice_size > 0
    num_slices = int(math.floor(size/slice_size))
    size_remainder = size - num_slices*slice_size
    assert size_remainder >= 0
    slices = [vec[k*slice_size:k*slice_size+slice_size] for k in range(num_slices)]
    if size_remainder and keep_remainder:
        slices.append(vec[-(size_remainder):])
    return slices


def big_logloss(y, y_prob, labels):
    # calc logloss w/o kernel crashing

    losses_vec = []
    for part in get_partitions(list(range(len(y_prob))), slice_size=1000):
        i, j = part[0], part[-1]   
        losses_vec.append(log_loss(y[i:j], y_prob[i:j], labels=labels))
    return np.mean(losses_vec)

def log(workdir, what):
    ts = utc_log_ts()
    with open(f'{workdir}/work.log', 'a') as fd:
        fd.write(f'{ts}, {what}\n')


def save_libsvm(X, y=None, outpath=None):
    # source: https://stackoverflow.com/a/31468805
    with open(outpath, 'w') as f:
        for j in range(X.shape[0]):
            f.write(" ".join(
                      [str(int(y[j])) if y is not None else ''] + ["{}:{}".format(i, X[j][i]) 
                      for i in range(X.shape[1]) if X[j][i] != 0]) + '\n')

