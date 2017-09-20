import BagLearner as bl
import LinRegLearner as rl
import numpy as np

class InsaneLearner(object):
    def __init__(self, verbose=False):
        self._learners = [
            bl.BagLearner(learner=rl.LinRegLearner, kwargs={}, bags=20, boost=False, verbose=False)
            for _ in range(20)
        ]

    def author(self):
        return 'arx3'

    def addEvidence(self, Xtrain, Ytrain):
        [learner.addEvidence(Xtrain, Ytrain) for learner in self._learners]

    def query(self, Xtest):
        predictions = np.concatenate([learner.query(Xtest).reshape(-1, 1) for learner in self._learners], axis=1)
        return np.mean(predictions, axis=1) 