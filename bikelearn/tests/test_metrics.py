import json
import unittest

import bikelearn.classify as blc

class ProbaSortedTest(unittest.TestCase):

    def test_basic(self):

        from nose.tools import set_trace; set_trace()

        y_test = [25, 16, 16, 17, 16, 16, 16, 16, 16, 16, 16, 2, 25, 17, 16, 25, 25, 22, 17, 17]

        classes = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24, 25, 26]
        with open('bikelearn/tests/data/probas1.json') as fd:
            out_probabilities = json.load(fd)


        sorted_outputs = blc.get_sorted_predict_proba_predictions(out_probabilities, classes, k=5)

        pass

