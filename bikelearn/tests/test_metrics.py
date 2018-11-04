import numpy as np
import json
import unittest

import bikelearn.classify as blc
import bikelearn.tests.utils as bltu
import bikelearn.metrics_utils as blmu

import bikelearn.settings as s

class ProbaSortedTest(unittest.TestCase):

    def test_basic(self):

        y_test = [25, 16, 16, 17, 16, 16, 16, 16, 16, 16, 16, 2, 25, 17, 16, 25, 25, 22, 17, 17]

        classes = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24, 25, 26]
        with open('bikelearn/tests/data/probas1.json') as fd:
            out_probabilities = json.load(fd)


        sorted_outputs = blmu.get_sorted_predict_proba_predictions(out_probabilities, classes, k=5)

        pass

    def test_predict_matches_proba(self):

        bundle, datasets, stations_df = bltu.make_basic_minimal_model()
        df = datasets['holdout_df']
        clf = bundle['clf']


        prepared = blc.predict_prepare(bundle, df, stations_df, labeled=True)

        y_topk1_outputs = blmu.get_sorted_predict_proba_predictions(
                prepared['y_predict_proba'],
                clf.classes_, k=1)

        from nose.tools import set_trace; set_trace()

        y_predictions, y_test, metrics = blc.run_model_predict(
               bundle, df, stations_df, labeled=True)



        assert [x[0] for x in y_topk1_outputs] == list(y_predictions)


