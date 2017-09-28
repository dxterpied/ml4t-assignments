"""Unit tests for gen_data.py

   It generates 15 datasets for each test case
   and validates that the best learner's RMSE
   is less than 90% of the worst learner's RMSE
"""

import DTLearner as dt
import LinRegLearner as lr
import math
import numpy as np
import unittest

from gen_data import best4LinReg, best4DT

class GenDataTest(unittest.TestCase):
    def test_best_for_LR_learner(self):
        datasets = self._generate_data_sets(best4LinReg)

        for (X, Y) in datasets:
            self._do_test(X, Y, lr.LinRegLearner(), dt.DTLearner(leaf_size=1))

    def test_best_for_DT_learner(self):
        datasets = self._generate_data_sets(best4DT)

        for (X, Y) in datasets:
            self._do_test(X, Y, dt.DTLearner(leaf_size=1), lr.LinRegLearner())

    def _generate_data_sets(self, generator):
        return [generator(seed=i*i) for i in range(15)]

    def _do_test(self, X, Y, best_learner, worst_learner):
        trainX, trainY, testX, testY = self._split_data_set(X, Y)

        best_learner.addEvidence(trainX, trainY)
        worst_learner.addEvidence(trainX, trainY)

        best_learner_rmse = self._rmse(testY, best_learner.query(testX))
        worst_learner_rmse = self._rmse(testY, worst_learner.query(testX))

        self._compare_rmse(best_learner_rmse, worst_learner_rmse)

    def _split_data_set(self, X, Y):
        # 60% for training data
        train_rows = int(math.floor(0.6* X.shape[0]))
        test_rows = X.shape[0] - train_rows
        
        # Randomly split into training and test sets
        train = np.random.choice(X.shape[0], size=train_rows, replace=False)
        test = np.setdiff1d(np.array(range(X.shape[0])), train)
        trainX = X[train, :] 
        trainY = Y[train]
        testX = X[test, :]
        testY = Y[test]

        return trainX, trainY, testX, testY

    def _rmse(self, targets, predictions):
        return math.sqrt(np.mean((targets - predictions) ** 2))

    def _compare_rmse(self, best_learner_rmse, worst_learner_rmse):
        self.assertLess(best_learner_rmse, 0.9 * worst_learner_rmse)

if __name__ == '__main__':
    unittest.main(verbosity=2)