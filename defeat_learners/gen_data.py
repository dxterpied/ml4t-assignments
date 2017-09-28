"""
template for generating data to fool learners (c) 2016 Tucker Balch
"""

import numpy as np
import math

def best4DT(seed=1489683273):
    np.random.seed(seed)
    coeffs = np.random.random(7)

    X = np.random.random(size=(1000, 2))
    # To fool the LinearRegression learner, we construct a
    # 2-variable polynomial which cannot be fit by a line.
    # A DecisionTree (regression) will have a better
    # chance of fitting the data more properly
    Y = coeffs[0] * X[:, 0] ** 2 * X[:, 1] ** 2 + \
        coeffs[1] * X[:, 0] ** 2 * X[:, 1] + \
        coeffs[2] * X[:, 0] * X[:, 1] ** 2 + \
        coeffs[3] * X[:, 0] * X[:, 1] + \
        coeffs[4] * X[:, 0] + \
        coeffs[5] * X[:, 1] + \
        coeffs[6]

    return X, Y

def best4LinReg(seed=1489683273):
    np.random.seed(seed)
    slope = np.random.random()
    intercept = np.random.random()

    X = np.random.random(size=(1000,2))
    # The best data set that can be generated for linear regression
    # is made of points that follow an underlying line!
    # In this case, we construct a line made of:
    # - A random slope
    # - A random intercept
    # - Points that are the result of adding up two random columns
    Y = (slope * X).sum(axis=1) + intercept

    return X, Y

def author():
    return 'arx3'

if __name__=="__main__":
    print "they call me Tim."
