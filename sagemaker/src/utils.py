from __future__ import print_function
import json
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
    stations_fn = get_bundle_stations_filename()
    if stations_fn:
        return os.path.join(model_path, stations_fn)

    return during_training_stations_filename()


def during_training_stations_filename():
    print("DEBUG, during training, stations_dir, " + stations_dir)
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


def get_bundle_stations_filename():
    out = get_bundle_meta_dict()
    return out.get('stations_filename')


def get_bundle_filename():
    out = get_bundle_meta_dict()
    return out.get('bundle_filename')


def get_bundle_meta_dict():
    path = os.path.join(model_path, 'bundle_meta.json')
    print('DEBUG get_bundle_filename, path, {}'.format(path))

    with open(path) as fd:
        out = json.load(fd)

    print('DEBUG, bundle_meta.json, {}'.format(out))
    assert out is not None, 'out, {}'.format(out)
    return out


def validate_stations_being_used(bundle, stations_id):
    bundle_stations_fn = bundle['train_metadata']['stations_df_fn']
    print('DEBUG, '
            'bundle_stations_fn, {} stations_id, {}'.format(
                bundle_stations_fn, stations_id)
            )
    assert (bundle_stations_fn.split('/')[-1]
            == stations_id.split('/')[-1])

