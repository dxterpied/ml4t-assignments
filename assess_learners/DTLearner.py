"""
Implements a Decision Tree learner
"""

import numpy as np

class DTLearner(object):
    def __init__(self, leaf_size=1, verbose=False):
        self._leaf_size = leaf_size
        self._verbose = verbose
        self._model = None

    def author(self):
        return 'arx3'

    def addEvidence(self, Xtrain, Ytrain):
        self._model = self._build_tree(Xtrain, Ytrain)

    def query(self, Xtest):
        return np.apply_along_axis(self._traverse_tree, 1, Xtest)

    def _build_tree(self, Xtrain, Ytrain, node=0):
        if Xtrain.shape[0] <= self._leaf_size:
            return self._build_leaf(node, np.mean(Ytrain))

        if self._same_label(Ytrain):
            return self._build_leaf(node, Ytrain[0])

        feature = self._determine_best_split(Xtrain, Ytrain)
        split_val = np.median(Xtrain[:, feature])

        splits = Xtrain[:, feature] <= split_val

        # Edge case: if splitting on the median causes all data to go
        # to a single branch, then that branch must become a leaf
        # because recursing would just go into an infinite loop
        # See: https://www.reddit.com/r/cs7646_fall2017/comments/704kdk/project_3_megathread_assess_learners/dn414lk/
        if self._same_split(splits):
            return self._build_leaf(node, np.mean(Ytrain))

        left_tree = self._build_tree(Xtrain[splits, :], Ytrain[splits], node=node+1)

        # We use left_tree.shape[0] + 1 + node to calculate the index of the node
        # for the right tree. We use the size of the built left tree, we add one
        # because we need to take into account the current node and finally we
        # add the current node's index to obtain the *absolute* value for the index
        # of the right tree node.
        right_tree = self._build_tree(Xtrain[~splits, :], Ytrain[~splits], node=left_tree.shape[0] + 1 + node)

        root = self._build_node(node, feature, split_val, node + 1, left_tree.shape[0] + 1 + node)

        return np.concatenate([root, left_tree, right_tree])

    def _build_leaf(self, node_index, node_value):
        return self._build_node(node_index, -1, node_value, -1, -1)

    def _build_node(self, node_index, feature, node_value, left_index, right_index):
        return np.array([node_index, feature, node_value, left_index, right_index]).reshape(1, -1)

    def _same_label(self, Ytrain):
        return np.unique(Ytrain).shape[0] == 1

    def _determine_best_split(self, Xtrain, Ytrain):
        # We use rowvar because each column represents a variable
        # and the rows, observations
        # See: https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html
        correlation_matrix = np.corrcoef(Xtrain, Ytrain, rowvar=False)

        # We need to remove the last row (-1) because it corresponds to the label Y
        # which we don't want to include in our calculation of the most correlated
        # feature. Also, we use the last column (-1) because it corresponds to the
        # correlation coefficient between features and the label
        return np.argmax(np.abs(correlation_matrix[0:-1, -1]), axis=0)

    def _same_split(self, splits):
        # Either goes to the left (all true)
        # or goes to the right (all false, negated, become all true)
        return np.alltrue(splits) or np.alltrue(~splits)

    def _traverse_tree(self, sample):
        feature_index = 1
        split_index = 2
        left_index = 3
        right_index = 4

        row_node = self._model[0, :]

        while True:
            feature = int(row_node[feature_index])
            split_val = row_node[split_index]

            if sample[feature] <= split_val:
                node = row_node[left_index]
            else:
                node = row_node[right_index]

            row_node = self._model[int(node), :]

            if row_node[feature_index] == -1:
                # For leaf nodes, the split value
                # is the actual node value
                return row_node[split_index]