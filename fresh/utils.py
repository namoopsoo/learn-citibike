import datetime
from collections import Counter
import pytz
import subprocess
import math
import pytz
import os
import numpy as np
from sklearn.metrics import log_loss
from joblib import Parallel, delayed, load, dump


def utc_ts():
    return datetime.datetime.utcnow(
        ).replace(tzinfo=pytz.UTC).strftime('%Y-%m-%dT%H%M%SZ')

def utc_log_ts():
    return datetime.datetime.utcnow(
        ).replace(tzinfo=pytz.UTC).strftime('%Y-%m-%d %H:%M:%SZ')

def do_walltime(start):
    return (datetime.datetime.now() - start).total_seconds()

    
def make_work_dir():
    ts = utc_ts()
    workdir = f'/opt/program/artifacts/{ts}' 
    os.mkdir(workdir)
    return workdir

def prepare_data(tripsdf, stationsdf):
    
    # Step 1, merge w/ stationsdf to get neighborhood data
    mdf = tripsdf[['start station name', 'end station name', 'gender',
                   'starttime', 'usertype']
            ].merge(stationsdf[['station_name', 'neighborhood']], 
                    left_on='start station name',
                    right_on='station_name'
                   ).rename(columns={'neighborhood': 'start_neighborhood'}
                           ).merge(stationsdf[['station_name', 'neighborhood']],
                                  left_on='end station name',
                                   right_on='station_name'
                                  ).rename(columns={'neighborhood': 'end_neighborhood'})
    
    neighborhoods = sorted(stationsdf.neighborhood.unique().tolist())

    prepare_weekday_feature(mdf)
    time_of_day_feature(mdf)
    
    X, y = (mdf[['start_neighborhood', 'gender', 'time_of_day', 'usertype', 'weekday', ]].values, 
            np.array(mdf['end_neighborhood'].tolist()))
    return X, y, neighborhoods


def prepare_weekday_feature(df):
    df['start_dt'] = df['starttime'].map(
                lambda x: datetime.datetime.strptime(
                    x, '%Y-%m-%d %H:%M:%S'
                    ).replace(tzinfo=pytz.timezone('US/Eastern')))
    weekdays = set(['Monday', 'Tuesday', 'Wednesday', 
                    'Thursday', 'Friday'])
    df['weekday'] = df['start_dt'].map(
                lambda x: int(x.strftime('%A') in weekdays))


def time_of_day_feature(df):
    hours = {'06': 0, '07': 0, '08': 0, '09': 0,
         '10': 1, '11': 1, '12': 1, '13': 1,
         '14': 2, '15': 2, '16': 2,
         '17': 3, '18': 3, '19': 3, '20': 3, '21': 3,
         '22': 4, '23': 4, '00': 4, '01': 4, '02': 4, '03': 4, '04': 4, '05': 4 }
    df['time_of_day'] = df['starttime'].map(lambda x: hours.get(x[11:13]))


def get_partitions(vec, slice_size, keep_remainder=True):
    size = _size(vec) 
    assert slice_size > 0
    num_slices = int(math.floor(size/slice_size))
    size_remainder = size - num_slices*slice_size
    assert size_remainder >= 0
    slices = [vec[k*slice_size:k*slice_size+slice_size] for k in range(num_slices)]
    if size_remainder and keep_remainder:
        slices.append(vec[-(size_remainder):])
    return slices


def get_slices(vec, slice_size=None, num_slices=None, keep_remainder=True):
    if num_slices:
        size = _size(vec) 
        slice_size = size//num_slices
    return [[part[0], part[-1] + 1] 
            for part in get_partitions(vec, slice_size, keep_remainder)]


def _size(vec):
    try:
        size = len(vec)
    except:
        size = vec.shape[0]
    return size

def big_logloss(y, y_prob, labels, parallel=True):
    # calc logloss w/o kernel crashing

    if parallel:
        losses_vec = Parallel(n_jobs=5)(delayed(log_loss)(y[part[0]:part[-1]], 
                                                   y_prob[part[0]:part[-1]], 
                                                   labels=labels) 
                                                   for part in get_partitions(list(range(len(y_prob))), 
                                                                              slice_size=1000))
        return np.mean(losses_vec)

    # else..
    losses_vec = []
    for part in get_partitions(list(range(len(y_prob))), slice_size=1000):
        i, j = part[0], part[-1]   
        losses_vec.append(log_loss(y[part[0]:part[-1]], y_prob[part[0]:part[-1]], labels=labels))
    return np.mean(losses_vec)

def _predict_worker(X, bundle_loc):
    bundle = load(bundle_loc)
    return bundle['model'].predict_proba(X)

def predict_proba(X, bundle_loc, slice_size=1000, parallel=True):
    if parallel:
        # chop it up
        slices = get_slices(list(range(X.shape[0])), slice_size=slice_size)
        vec = Parallel(n_jobs=5)(delayed(_predict_worker)(X[a:b], bundle_loc)
                                                         for (a, b) in slices)
        return np.concatenate(vec)
    else:
        return _predict_worker(X, bundle_loc)


def log(workdir, *what):
    ts = utc_log_ts()
    stuff = ', '.join(what)
    with open(f'{workdir}/work.log', 'a') as fd:
        fd.write(f'{ts}, {stuff}\n')


def save_libsvm(X, y=None, outpath=None):
    # source: https://stackoverflow.com/a/31468805
    with open(outpath, 'w') as f:
        for j in range(X.shape[0]):
            f.write(" ".join(
                      [str(int(y[j])) if y is not None else ''] + ["{}:{}".format(i, X[j][i]) 
                      for i in range(X.shape[1]) if X[j][i] != 0]) + '\n')


def get_my_memory():
    mypid = os.getpid()
    out = subprocess.check_output(["ps", "-p", f"{mypid}", "-o", "pid,ppid,pmem,rss"])

    pid, ppid, pmem, rss = out.decode('utf-8').split('\n')[1].strip().split()
    # print(f'{pid}, {pmem}, {rss}')
    gigs = int(rss)/1024/1024
    assert int(pid) == mypid
    return {'pmem': pmem, 'rss': f'{round(gigs, 3)} GiB'}

def rebalance_proportions(proportions):

    # a = np.array([.1, .2, .3, .4])
    assert sum(proportions) == 1.

    average = 1/proportions.shape[0]

    b = average/proportions
    new_norm = np.sum(b)
    c = b/new_norm
    return c

def balance_dataset(X, y):
    # Balance before encoding

    dd = dict(Counter(y)).items()
    vv = rebalance_proportions()
'''
X, y, neighborhoods = fu.prepare_data(tripsdf, stationsdf)

X, y = balance_dataset(X, y)

'''
