# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
import StringIO
import sys
import signal
import traceback

import flask

import pandas as pd

import bikelearn.classify as blc
import bikelearn.settings as s

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.


def get_bundle_filename():
    path = os.path.join(model_path, 'bundle_meta.json')
    print('DEBUG get_bundle_filename, path, {}'.format(path))

    with open(path) as fd:
        out = json.load(fd)

    print('DEBUG, bundle_meta.json, {}'.format(out))
    assert out is not None, 'out, {}'.format(out)
    return out.get('bundle_filename')


def do_predict(bundle, df):
    stations_df = bundle['train_metadata']['stations_df']

    widened_df = blc.widen_df_with_other_cols(df, s.ALL_COLUMNS)

    print('DEBUG df.shape, ' + str(df.shape))
    print('DEBUG widened_df.shape, ' + str(widened_df.shape))

    y_predictions, _ = blc.run_model_predict(
            bundle, widened_df, stations_df, labeled=False)

    return y_predictions


class ScoringService(object):
    # model = None                # Where we keep the model when it's loaded
    bundle = None

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.bundle is None:
            bundle_filename = get_bundle_filename()
            with open(os.path.join(model_path, bundle_filename), 'r') as inp:
                cls.bundle = pickle.load(inp)
        return cls.bundle

    @classmethod
    def predict(cls, csvdata):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        bundle = cls.get_model()
        print ('DEBUG, csvdata, {}'.format(csvdata))
        df = blc.hydrate_csv_to_df(csvdata)
        print('Invoked with {} records'.format(df.shape[0]))

        print ('DEBUG, ')
        df.head()

        preds = do_predict(bundle, df)
        print ('DEBUG, preds, {}'.format(preds))

        return preds

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 500 # why was this 404 prior?
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None
    print('DEBUG, flask.request.content_type, "{}"'.format(flask.request.content_type))

    # Convert from CSV to pandas
    if flask.request.content_type == 'text/csv':
        print ('DEBUG, ok text/csv')
        data = flask.request.data.decode('utf-8')


        # FIXME ok eventually put back the header=None 
        # sio = StringIO.StringIO(data)
        # data = pd.read_csv(sio, header=None)

    else:
        print ('DEBUG, hmm, not text/csv')
        return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')


    # Do the prediction
    predictions = ScoringService.predict(data)

    # Convert from numpy back to CSV
    out = StringIO.StringIO()
    pd.DataFrame({'results': predictions}).to_csv(out, header=False, index=False)
    result = out.getvalue()

    return flask.Response(response=result, status=200, mimetype='text/csv')


