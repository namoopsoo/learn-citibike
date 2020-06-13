import datetime
import pytz
import os
import numpy as np

def utc_ts():
    return datetime.datetime.utcnow(
        ).replace(tzinfo=pytz.UTC).strftime('%Y-%m-%dT%H%M%SZ')
    
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
    
