"""Contains code for running and generating charts for Experiment 2"""

__author__ = "Allan Reyes"
__email__ = "reyallan@gatech.edu"
__userId__ = "arx3"

import datetime as dt
import json
import logging
import numpy as np
import pandas as pd
import random
import sys

from marketsimcode import compute_portvals
from StrategyLearner import StrategyLearner

def main():
    """
    Provide an hypothesis regarding how changing the value of impact
    should affect in sample trading behavior and results (provide at
    least two metrics). Conduct an experiment with JPM on the
    in sample period to test that hypothesis. Provide charts, graphs
    or tables that illustrate the results of your experiment.
    """

    # Hypothesis:
    # The `impact` encapsulates the volatility, stability and overall
    # fluctuation of the market; in particular, movements that would
    # affect one's portfolio, e.g. unexpected (i.e. not predicted)
    # increases or drops in prices.
    # For the StrategyLearner should directly affect the learned
    # policy, particularly, in terms of willingness to take risks by
    # betting on the behavior of the market.
    # This can be translated into three metrics:
    # - Number of entries:
    #   These should be reduced as market impact increases which
    #   shows the learning agent being more cautious about its bets
    # - Cumulative return:
    #   Directly related to the point mentioned above, as market
    #   impact increases and the agent's willingness to take risks
    #   decreaes, so is the overall performance of the strategy
    # - Training episodes:
    #   This applies specifically to the Q-Learning agent, but it
    #   is interesting to see how as the market impact increases,
    #   the number of complete training episodes (i.e. a complete
    #   pass on the trading data) is not affected. One would think
    #   that the agent would converge faster when the impact is
    #   large as it would quickly realize that the most optimal
    #   strategy is to not do anything. However, impact does not
    #   affect the rate of convergence, but rather the strategy
    #   that the agent converges to

    # Set the seed for reproducibility
    random.seed(1481090000)

    # Experiment parameters
    symbol = 'JPM'
    # In-sample: January 1, 2008 to December 31 2009
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    starting_value = 100000
    commission = 0.0
    # Values to use to evaluate the effect of the impact
    impact_values = [0.0, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]

    all_entries = []
    all_returns = []
    all_episodes = []

    for impact in impact_values:
        log.info("Evaluating the effect of impact=%s", impact)
        strategy_learner = StrategyLearner(verbose=False, impact=impact)

        log.info("Training StrategyLearner")
        strategy_learner.addEvidence(
            symbol=symbol,
            sd=start_date,
            ed=end_date,
            sv=starting_value
        )

        log.info("Querying StrategyLearner to generate trades")
        trades = strategy_learner.testPolicy(
            symbol=symbol,
            sd=start_date,
            ed=end_date,
            sv=starting_value
        )

        log.info("Transforming StrategyLearner trades into marketsim orders")
        orders = _convert_trades_to_marketisim_orders(symbol, trades)

        log.info("Computing portfolio values for %d orders", orders.shape[0])
        port_vals = compute_portvals(
            orders,
            start_val=starting_value,
            commission=commission,
            impact=impact
        )

        cumulative_return = _compute_cumulative_return(port_vals)

        all_entries.append(strategy_learner.metadata['entries'])
        all_returns.append(cumulative_return)
        all_episodes.append(strategy_learner.metadata['training_episodes'])

    _plot_and_save_number_of_entries_per_impact_value(impact_values, all_entries)
    _plot_and_save_number_of_episodes_per_impact_value(impact_values, all_episodes)
    _plot_and_save_cumulative_return_per_impact_value(impact_values, all_returns)

def _plot_and_save_number_of_entries_per_impact_value(impact_values, entries):
    _generate_bar_plot(
        entries,
        'Number of entries per impact value - In Sample',
        'Impact value',
        'Number of entries',
        'Entries',
        impact_values,
        'Experiment2-NumberOfEntries'
    )

    _save_as_json(impact_values, entries, 'entries_per_impact')

def _plot_and_save_number_of_episodes_per_impact_value(impact_values, episodes):
    _generate_bar_plot(
        episodes,
        'Number of training episodes per impact value - In Sample',
        'Impact value',
        'Number of training episodes',
        'Episodes',
        impact_values,
        'Experiment2-NumberOfEpisodes'
    )

    _save_as_json(impact_values, episodes, 'episodes_per_impact')

def _plot_and_save_cumulative_return_per_impact_value(impact_values, returns):
    _generate_bar_plot(
        returns,
        'Cumulative return per impact value - In Sample',
        'Impact value',
        'Cumulative return (%)',
        'Cumulative return',
        impact_values,
        'Experiment2-CumulativeReturn'
    )

    _save_as_json(impact_values, returns, 'cumulative_return_per_impact')

def _convert_trades_to_marketisim_orders(symbol, trades):
    # Convert the trades into the format expected by my marketsimcode.py
    orders = pd.DataFrame(index=trades.index, columns=['Order', 'Date', 'Symbol', 'Shares'])

    for index, trade in trades.iterrows():
        shares = trade['Shares']

        if shares == 0:
            orders.loc[index] = ['HOLD', index, symbol, shares]
        elif shares > 0:
            orders.loc[index] = ['BUY', index, symbol, shares]
        else:
            orders.loc[index] = ['SELL', index, symbol, shares * -1]

    return orders

def _compute_cumulative_return(portfolio_values):
    return (portfolio_values[-1] / portfolio_values[0]) - 1

def _generate_bar_plot(data, title, xlabel, ylabel, bar_label, groups, filename):
    # See: https://matplotlib.org/examples/pylab_examples/barchart_demo.html
    log.info("Generating bar plot with title '%s'", title)

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    plt.close()

    _, ax = plt.subplots()
    ax.grid(color='black', linestyle='dotted')

    index = np.arange(len(groups))
    bar_width = 0.35
    opacity = 0.4

    bar = plt.bar(index, data, bar_width, alpha=opacity, color='b', label=bar_label)

    plt.xlabel(xlabel, fontsize='15')
    plt.ylabel(ylabel, fontsize='15')
    plt.title(title)

    # Rotate tick labels and align them
    # See: https://stackoverflow.com/questions/14852821/aligning-rotated-xticklabels-with-their-respective-xticks
    # See: https://matplotlib.org/examples/ticks_and_spines/ticklabels_demo_rotation.html
    plt.xticks(index + bar_width / 2, groups, rotation=45, ha='right')

    plt.legend()
    plt.tight_layout()

    plt.savefig('report/img/{}.png'.format(filename), bbox_inches='tight')

    log.info("Saved bar plot to file: %s", filename)

def _save_as_json(impact_values, metrics, filename):
    log.info("Creating JSON data")

    data = {}
    for index, impact in enumerate(impact_values):
        data[impact] = metrics[index]

    filepath = './report/json/{}.json'.format(filename)
    with open(filepath, 'w') as outfile:
        json.dump(data, outfile, indent=2, separators=(',', ': '))

    log.info("JSON data saved to file: %s", filename)

if __name__ == '__main__':
    # Configure our logger
    log = logging.getLogger(__name__)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    )
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False

    main()