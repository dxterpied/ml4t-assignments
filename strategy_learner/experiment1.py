"""Contains code for running and generating charts for Experiment 1"""

__author__ = "Allan Reyes"
__email__ = "reyallan@gatech.edu"
__userId__ = "arx3"

import datetime as dt
import logging
import pandas as pd
import random
import sys

from ManualStrategy import ManualStrategy
from marketsimcode import compute_portvals
from StrategyLearner import StrategyLearner

def main():
    """
    Using exactly the same indicators that you used in manual_strategy,
    compare your manual strategy with your learning strategy in sample.
    Plot the performance of both strategies in sample along with the benchmark.
    Trade only the symbol JPM for this evaluation.
    """

    # Set the seed for reproducibility
    # random.seed(1481090000)

    # Experiment parameters
    params = {
        'symbol': 'JPM',
        # In-sample: January 1, 2008 to December 31 2009
        'start_date': dt.datetime(2008, 1, 1),
        'end_date': dt.datetime(2009, 12, 31),
        'starting_value': 100000,
        'commission': 0.0,
        'impact': 0.0
    }

    manual_strategy_values = _evaluate_manual_strategy(params)
    strategy_learner_values = _evaluate_strategy_learner(params)
    benchmark_values = _evaluate_benchmark(params, strategy_learner_values.index)

    # Normalize values
    manual_strategy_values = _normalize(manual_strategy_values)
    strategy_learner_values = _normalize(strategy_learner_values)
    benchmark_values = _normalize(benchmark_values)

    trading_dates = strategy_learner_values.index

    _plot(
        [
            (trading_dates, strategy_learner_values),
            (trading_dates, manual_strategy_values),
            (trading_dates, benchmark_values)
        ],
        'StrategyLearner vs ManualStrategy vs Benchmark - In Sample',
        'Date',
        'Normalized Value',
        'Experiment1',
        colors=['blue', 'red', 'black'],
        legend=['StrategyLearner', 'ManualStrategy', 'Benchmark']
    )

def _evaluate_manual_strategy(params):
    log.info("Evaluating ManualStrategy using params: %s", params)

    symbol = params['symbol']
    start_date = params['start_date']
    end_date = params['end_date']
    starting_value = params['starting_value']
    commission = params['commission']
    impact = params['impact']

    manual_strategy = ManualStrategy()

    log.info("Querying ManualStrategy to generate orders")
    orders = manual_strategy.testPolicy(
        symbol=symbol,
        sd=start_date,
        ed=end_date,
        sv=starting_value
    )

    log.info("Computing portfolio values for %d orders", orders.shape[0])
    port_vals = compute_portvals(
        orders,
        start_val=starting_value,
        commission=commission,
        impact=impact
    )

    cumulative_return = _get_portfolio_performance(port_vals)

    log.info("ManualStrategy stats: final value=%s, cumulative return=%s",
        port_vals.iloc[-1], cumulative_return
    )

    return port_vals

def _evaluate_strategy_learner(params):
    log.info("Evaluating StrategyLearner using params: %s", params)

    symbol = params['symbol']
    start_date = params['start_date']
    end_date = params['end_date']
    starting_value = params['starting_value']
    commission = params['commission']
    impact = params['impact']

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

    cumulative_return = _get_portfolio_performance(port_vals)

    log.info("StrategyLearner stats: final value=%s, cumulative return=%s",
        port_vals.iloc[-1], cumulative_return
    )

    return port_vals

def _evaluate_benchmark(params, trading_dates):
    log.info("Evaluating benchmark using params: %s", params)

    symbol = params['symbol']
    starting_value = params['starting_value']
    commission = params['commission']
    impact = params['impact']

    log.info("Generating orders for benchmark")
    orders = pd.DataFrame(index=trading_dates, columns=['Order', 'Date', 'Symbol', 'Shares'])
    # Buy and hold position
    orders.iloc[0] = ['BUY', trading_dates[0], symbol, 1000]
    orders.iloc[1:] = [['HOLD', date, symbol, 0] for date in trading_dates[1:]]

    log.info("Computing portfolio values for %d orders", orders.shape[0])
    port_vals = compute_portvals(
        orders,
        start_val=starting_value,
        commission=commission,
        impact=impact
    )

    cumulative_return = _get_portfolio_performance(port_vals)

    log.info("Benchmark stats: final value=%s, cumulative return=%s",
        port_vals.iloc[-1], cumulative_return
    )

    return port_vals

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

def _get_portfolio_performance(portfolio_values):
    # We are measuring performance in terms of the cumulative return
    return (portfolio_values[-1] / portfolio_values[0]) - 1

def _normalize(values):
    return values / values.iloc[0]

def _plot(data, title, xlabel, ylabel, filename, colors=['b', 'r', 'g'], legend=None):
    log.info("Generating plot with title '%s'", title)

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    plt.close()

    fig, ax = plt.subplots()
    # Fix display issues when using dates as x-labels
    # https://matplotlib.org/users/recipes.html
    fig.autofmt_xdate()

    ax.grid(color='black', linestyle='dotted')
    # Display dates every 2 months
    # https://matplotlib.org/users/recipes.html
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    [plt.plot(
        xdata,
        ydata,
        linewidth=2.5,
        color=colors[index],
        alpha=0.4,
        label=legend[index])
     for index, (xdata, ydata) in enumerate(data)]

    plt.title(title)
    plt.xlabel(xlabel, fontsize='15')
    plt.ylabel(ylabel, fontsize='15')

    if not legend is None:
        plt.legend(fontsize='small')

    plt.savefig("report/img/{}.png".format(filename), bbox_inches='tight')

    log.info("Saved plot to file: %s", filename)

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