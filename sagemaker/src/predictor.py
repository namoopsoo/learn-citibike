# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.


import os
import json
import pickle
import sys
import signal
import traceback
import joblib
import flask

import pandas as pd

# import bikelearn.classify as blc
# import bikelearn.settings as s
# import utils as ut
import fresh.utils as fu
import fresh.predict_utils as fpu


from io import StringIO

bundle_loc = '/opt/program/artifacts/2020-08-19T144654Z/all_bundle.joblib'
bundle_loc = '/opt/program/artifacts/2020-08-19T144654Z/all_bundle_with_stationsdf.joblib'
print('bundle_loc', bundle_loc)

bundle = fpu.load_bundle(bundle_loc)

record = {
 'starttime': '2013-07-01 00:00:00',
 'start station id': 164,
 'start station name': 'E 47 St & 2 Ave',
 'start station latitude': 40.75323098,
 'start station longitude': -73.97032517,
# unknown
# 'end station id': 504,
# 'end station name': '1 Ave & E 15 St',
# 'end station latitude': 40.73221853,
# 'end station longitude': -73.98165557,
# 'stoptime': '2013-07-01 00:10:34',
# 'tripduration': 634,
 'bikeid': 16950,
 'usertype': 'Customer',
 'birth year': '\\N',
 'gender': 0}


out = fpu.full_predict(bundle, record)
print('predict out', out)




# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.


def predict_or_validation(bundle, csvdata):
    if os.getenv('DO_VALIDATION'):
        return do_batch_validation(bundle, csvdata)

    return do_predict(bundle, csvdata)


def do_predict(bundle, csvdata):
    stations_fn, stations_df = ut.get_stations()
    ut.validate_stations_being_used(bundle, stations_fn)

    df = blc.hydrate_and_widen(csvdata)
    print('DEBUG predict, df.shape, ' + str(df.shape))

    y_predictions, _, _ = blc.run_model_predict(
            bundle, df, stations_df, labeled=False)

    return numpy_to_csv(y_predictions)


def do_batch_validation(bundle, csvdata):
    stations_fn, stations_df = ut.get_stations()
    ut.validate_stations_being_used(bundle, stations_fn)

    df = blc.hydrate_labeled_csvdata(csvdata)
    print('DEBUG batch_validation, df.shape, ' + str(df.shape))

    y_predictions, y_test, metrics = blc.run_model_predict(
            bundle, df, stations_df, labeled=True)

    return json.dumps(metrics)


class ScoringService(object):
    bundle = None

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.bundle is None:
            bundle_filename = ut.get_bundle_filename()
            with open(os.path.join(ut.model_path, bundle_filename), 'r') as inp:
                cls.bundle = pickle.load(inp)
        return cls.bundle

    @classmethod
    def predict(cls, csvdata):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        bundle = cls.get_model()
        print('DEBUG, csvdata[:1000], {}'.format(csvdata[:1000]))

        out = predict_or_validation(bundle, csvdata)
        print('DEBUG, out, {}'.format(out))
        return out

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    #health = ScoringService.get_model() is not None  # You can insert a health check here
    health = True

    status = 200 if health else 500 # why was this 404 prior?
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    csvdata = None
    request_content_type = flask.request.content_type
    print('DEBUG, flask.request.content_type, "{}"'.format(request_content_type))

    if request_content_type == 'text/csv':
        csvdata = flask.request.data.decode('utf-8')
        result = ScoringService.predict(csvdata=csvdata)
        do_validation = os.getenv('DO_VALIDATION', False)
        print('DEBUG, DO_VALIDATION env, ' + str(do_validation))
        mimetype = {False: 'text/csv',
                'yes': 'application/json'}[do_validation]
        return flask.Response(response=result, status=200, mimetype=mimetype)


    print ('DEBUG, hmm, not text/csv or application/json')
    return flask.Response(response='This predictor only supports CSV data',
            status=415, mimetype='text/plain')


def determine_response_type(querystring):
    response_type = querystring.get('response', 'simple')
    print('DEBUG, querystring, ', querystring)
    return response_type


def numpy_to_csv(predictions):
    out = StringIO.StringIO()
    pd.DataFrame({'results': predictions}).to_csv(out, header=False, index=False)
    result = out.getvalue()
    return result

