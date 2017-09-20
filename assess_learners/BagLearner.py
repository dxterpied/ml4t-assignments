"""
Implements a Bootstrap Aggregating Learner
"""

import numpy as np

class BagLearner(object):
    def __init__(self, learner=None, kwargs={}, bags=20, boost=False, verbose=False):
        self._learner = learner
        self._kwargs = kwargs
        self._bags = bags
        self._boost = boost
        self._verbose = verbose

        self._learners = self._create_learners()

    def author(self):
        return 'arx3'

    def addEvidence(self, Xtrain, Ytrain):
        for learner in self._learners:
            Xtrain_prime, Ytrain_prime = self._bootstrap(Xtrain, Ytrain)
            learner.addEvidence(Xtrain_prime, Ytrain_prime)

    def query(self, Xtest):
        # Make sure the predictions are column-vectors by reshaping them to be of size (N, 1)
        # and concatenating them horizontally, i.e. via the columns (axis=1)
        predictions = np.concatenate([learner.query(Xtest).reshape(-1, 1) for learner in self._learners], axis=1)
        # Aggregate the predictions of all learners, i.e. columns (axis=1) for all samples
        return np.mean(predictions, axis=1)

    def _create_learners(self):
        return [self._learner(**self._kwargs) for _ in range(self._bags)]

    def _bootstrap(self, Xtrain, Ytrain):
        # Sample N indices from the range(0, N)
        # See: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.choice.html
        indices = np.random.choice(Xtrain.shape[0], size=Xtrain.shape[0], replace=True)
        assert indices.shape[0] == Xtrain.shape[0]

        return Xtrain[indices, :], Ytrain[indices]