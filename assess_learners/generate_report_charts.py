"""
Generates all the charts for the report
"""

import BagLearner as bl
import DTLearner as dt
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import RTLearner as rt
import util

def main():
    # Set the seed for reproducibility
    np.random.seed(0)

    print "1. Does overfitting occur with respect to leaf_size?"
    _generate_overfitting_leaf_size_chart()

    print "2. Can bagging reduce or eliminate overfitting with respect to leaf_size?"
    _generate_overfitting_leaf_size_bagging_chart()

    print "3. Quantitatively compare DT and RT learners"
    _generate_rmse_comparison_dt_and_rt_chart()
    _generate_correlation_comparison_dt_and_rt_chart()
    _generate_overfitting_leaf_size_rt_chart()

def _generate_overfitting_leaf_size_chart():
    trainX, trainY, testX, testY = _create_train_and_test_sets(_read_data('Istanbul.csv'))

    # Try leaf sizes from 1 to 20
    leaf_sizes = range(1,21)
    out_of_sample = []
    in_sample = []

    for leaf_size in leaf_sizes:
        learner = dt.DTLearner(leaf_size=leaf_size)
        learner.addEvidence(trainX, trainY)

        in_sample.append(_rmse(learner.query(trainX), trainY))
        out_of_sample.append(_rmse(learner.query(testX), testY))

    _generate_plot(
        [(leaf_sizes, in_sample), (leaf_sizes, out_of_sample)],
        'Overfitting vs Leaf Size in Decision Tree',
        'Leaf Size',
        'RMSE',
        legend=['In Sample', 'Out of Sample']
    )

def _generate_overfitting_leaf_size_bagging_chart():
    trainX, trainY, testX, testY = _create_train_and_test_sets(_read_data('Istanbul.csv'))

    # Try leaf sizes from 1 to 20
    leaf_sizes = range(1,21)
    out_of_sample = []
    in_sample = []
    bags = 100

    for leaf_size in leaf_sizes:
        learner = bl.BagLearner(learner=dt.DTLearner, kwargs={'leaf_size':leaf_size}, bags=bags)
        learner.addEvidence(trainX, trainY)

        in_sample.append(_rmse(learner.query(trainX), trainY))
        out_of_sample.append(_rmse(learner.query(testX), testY))

    _generate_plot(
        [(leaf_sizes, in_sample), (leaf_sizes, out_of_sample)],
        'Overfitting vs Leaf Size with {} Bags'.format(bags),
        'Leaf Size',
        'RMSE',
        legend=['In Sample', 'Out of Sample']
    )

def _generate_rmse_comparison_dt_and_rt_chart():
    filenames = [
        '3_groups.csv',
        'Istanbul.csv',
        'ripple.csv',
        'simple.csv',
        'winequality-red.csv',
        'winequality-white.csv'
    ]

    out_of_sample_dt = []
    out_of_sample_rt = []
    leaf_size = 10

    for filename in filenames:
        trainX, trainY, testX, testY = _create_train_and_test_sets(_read_data(filename))
        
        learner = dt.DTLearner(leaf_size=leaf_size)
        learner.addEvidence(trainX, trainY)

        out_of_sample_dt.append(_rmse(learner.query(testX), testY))

        learner = rt.RTLearner(leaf_size=leaf_size)
        learner.addEvidence(trainX, trainY)

        out_of_sample_rt.append(_rmse(learner.query(testX), testY))

    _generate_pair_bar_plot(
        out_of_sample_dt,
        out_of_sample_rt,
        'Generalization: Decision Tree vs Random Tree',
        'Data',
        'Out-of-sample RMSE',
        filenames,
        ['Decision Tree', 'Random Tree']
    )

def _generate_correlation_comparison_dt_and_rt_chart():
    filenames = [
        '3_groups.csv',
        'Istanbul.csv',
        'ripple.csv',
        'simple.csv',
        'winequality-red.csv',
        'winequality-white.csv'
    ]

    corr_dt = []
    corr_rt = []
    leaf_size = 10

    for filename in filenames:
        trainX, trainY, testX, testY = _create_train_and_test_sets(_read_data(filename))
        
        learner = dt.DTLearner(leaf_size=leaf_size)
        learner.addEvidence(trainX, trainY)

        corr_dt.append(_correlation(learner.query(testX), testY))

        learner = rt.RTLearner(leaf_size=leaf_size)
        learner.addEvidence(trainX, trainY)

        corr_rt.append(_correlation(learner.query(testX), testY))

    _generate_pair_bar_plot(
        corr_dt,
        corr_rt,
        'Correlation: Decision Tree vs Random Tree',
        'Data',
        'Correlation',
        filenames,
        ['Decision Tree', 'Random Tree']
    )

def _generate_overfitting_leaf_size_rt_chart():
    trainX, trainY, testX, testY = _create_train_and_test_sets(_read_data('Istanbul.csv'))

    # Try leaf sizes from 1 to 20
    leaf_sizes = range(1,21)
    out_of_sample = []
    in_sample = []

    for leaf_size in leaf_sizes:
        learner = rt.RTLearner(leaf_size=leaf_size)
        learner.addEvidence(trainX, trainY)

        in_sample.append(_rmse(learner.query(trainX), trainY))
        out_of_sample.append(_rmse(learner.query(testX), testY))

    _generate_plot(
        [(leaf_sizes, in_sample), (leaf_sizes, out_of_sample)],
        'Overfitting vs Leaf Size in Random Tree',
        'Leaf Size',
        'RMSE',
        legend=['In Sample', 'Out of Sample']
    )

def _read_data(filename):
    with util.get_learner_data_file(filename) as f:
        alldata = np.genfromtxt(f, delimiter=',')

        if filename == 'Istanbul.csv':
            # Skip the date column (first) and header row (first)
            return alldata[1:, 1:]

        return alldata

def _create_train_and_test_sets(alldata):
    # 60% for training
    cutoff = int(alldata.shape[0] * 0.6)

    train_data = alldata[:cutoff, :]
    trainX = train_data[:, :-1]
    trainY = train_data[:, -1]
    
    test_data = alldata[cutoff: ,:]
    testX = test_data[:, :-1]
    testY = test_data[:, -1]

    return trainX, trainY, testX, testY

def _rmse(predictions, targets):
    return math.sqrt(np.mean((targets - predictions) ** 2))

def _correlation(predictions, targets):
    # The first row and second colum (0, 1) contains the
    # correlation of the predictions with the targets
    return np.corrcoef(predictions, y=targets)[0, 1]

def _generate_plot(data, title, xlabel, ylabel, legend=None):
    plt.close()

    _, ax = plt.subplots()

    # Format the x-axis so that ticks are shown as integers
    # See: https://www.scivision.co/matplotlib-force-integer-labeling-of-axis/
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Also format the x-axis to have ticks displayed with a step of 1.0
    # See: https://stackoverflow.com/questions/12608788/changing-the-tick-frequency-on-x-or-y-axis-in-matplotlib
    loc = ticker.MultipleLocator(base=1)
    ax.xaxis.set_major_locator(loc)

    ax.grid(color='black', linestyle='dotted')

    colors = ['b', 'r', 'g']
    [plt.plot(xdata, ydata, linewidth=2.5, color=colors[index], alpha=0.4) for index, (xdata, ydata) in enumerate(data)]

    plt.title(title)
    plt.xlabel(xlabel, fontsize='15')
    plt.ylabel(ylabel, fontsize='15')

    if not legend is None:
        plt.legend(legend)

    plt.savefig('report/img/{}.png'.format(title.replace(' ', '')), bbox_inches='tight')

def _generate_pair_bar_plot(data1, data2, title, xlabel, ylabel, groups, pair_labels):
    # See: https://matplotlib.org/examples/pylab_examples/barchart_demo.html
    plt.close()

    _, ax = plt.subplots()
    ax.grid(color='black', linestyle='dotted')

    index = np.arange(len(groups))
    bar_width = 0.35
    opacity = 0.4

    bar1 = plt.bar(index, data1, bar_width, alpha=opacity, color='b', label=pair_labels[0])
    bar2 = plt.bar(index + bar_width, data2, bar_width, alpha=opacity, color='r', label=pair_labels[1])
    
    plt.xlabel(xlabel, fontsize='15')
    plt.ylabel(ylabel, fontsize='15')
    plt.title(title)

    # Rotate tick labels and align them
    # See: https://stackoverflow.com/questions/14852821/aligning-rotated-xticklabels-with-their-respective-xticks
    # See: https://matplotlib.org/examples/ticks_and_spines/ticklabels_demo_rotation.html
    plt.xticks(index + bar_width / 2, groups, rotation=45, ha='right')

    plt.legend()
    plt.tight_layout()
    
    plt.savefig('report/img/{}.png'.format(title.replace(' ', '')), bbox_inches='tight')

if __name__ == '__main__':
    main()