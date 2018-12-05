from __future__ import print_function
import os
import pandas as pd


prefix = '/opt/ml/'
input_path = prefix + 'input/data'
output_path = os.path.join(prefix, 'output')
model_path = os.path.join(prefix, 'model')
param_path = os.path.join(prefix, 'input/config/hyperparameters.json')
config_path = os.path.join(prefix, 'input/config')

# This algorithm has a single channel of input data called 'training'. Since we run in
# File mode, the input files are copied to the directory specified here.
training_path = os.path.join(input_path, 'training')
testing_path = os.path.join(input_path, 'testing')
stations_dir = os.path.join(training_path, 'support')

def whats_stations_filename():
    print("DEBUG, stations_dir, " + stations_dir)
    stations_fn_list = [file
            for file in
            os.listdir(stations_dir)
            if file.endswith('.csv')]

    print('DEBUG, stations_fn_list , ' + str(stations_fn_list))
    assert len(stations_fn_list) == 1, 'should be one file here'
    stations_fn = stations_fn_list[0]
    return os.path.join(stations_dir, stations_fn)


def get_stations():
    stations_fn = whats_stations_filename()
    stations_df = pd.read_csv(stations_fn)

    return stations_fn, stations_df

