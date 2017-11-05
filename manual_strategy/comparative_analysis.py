"""
Evaluates the performance of the manual strategy
against the benchmark for the out-of-sample period
"""

import datetime as dt
import pandas as pd

from ManualStrategy import ManualStrategy
from marketsimcode import compute_portvals
from util import get_data

def main():
    """
    Generates the charts and metrics for the
    manual strategy and benchmark for the
    out-of-sample period
    """
    manual_port_vals, manual_orders, long_short_data = _evaluate_manual_stategy()
    benchmark_port_vals = _evaluate_benchmark(manual_port_vals.index)

    # Save the orders generated for information purposes
    manual_orders.to_csv('report/notes/manual_strategy_orders_out_of_sample.csv')

    # Normalize values
    manual_port_vals = manual_port_vals / manual_port_vals.iloc[0]
    benchmark_port_vals = benchmark_port_vals / benchmark_port_vals.iloc[0]

    trading_dates = manual_port_vals.index

    _plot(
        [
            (trading_dates, benchmark_port_vals),
            (trading_dates, manual_port_vals)
        ],
        long_short_data,
        'Manual Strategy vs Benchmark - Out of Sample',
        'Date',
        'Normalized Value',
        colors=['blue', 'black'],
        legend=['Benchmark', 'Manual Strategy']
    )

    # Also plot the prices for the out-of-sample error
    prices = get_data(['JPM'], pd.date_range(dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31)))
    _plot(
        [
            (trading_dates, prices['JPM'])
        ],
        [],
        'Prices for JPM - Out of Sample',
        'Date',
        'Price',
        legend=['Prices']
    )

def _evaluate_manual_stategy():
    ms = ManualStrategy()
    orders = ms.testPolicy(
        symbol="JPM",
        sd=dt.datetime(2010, 1, 1),
        ed=dt.datetime(2011, 12, 31),
        sv=100000
    )
    port_vals = compute_portvals(
        orders,
        start_val=100000,
        commission=9.95,
        impact=0.005
    )
    cumulative_return, std_dr, mean_dr = _get_portfolio_stats(port_vals)

    print "\n===== Manual Strategy Stats ====="

    print "Final portfolio value: ", port_vals.iloc[-1]
    print "Cumulative return: ", cumulative_return
    print "Std of daily returns: ", std_dr
    print "Mean of daily returns: ", mean_dr

    return port_vals, orders, ms.get_long_short_entries()

def _evaluate_benchmark(trading_dates):
    orders = pd.DataFrame(index=trading_dates, columns=['Order', 'Date', 'Symbol', 'Shares'])
    # Buy and hold position
    orders.iloc[0] = ['BUY', trading_dates[0], 'JPM', 1000]
    orders.iloc[1:] = [['HOLD', date, 'JPM', 0] for date in trading_dates[1:]]

    port_vals = compute_portvals(
        orders,
        start_val=100000,
        commission=9.95,
        impact=0.005
    )

    cumulative_return, std_dr, mean_dr = _get_portfolio_stats(port_vals)

    print "\n===== Benchmark Strategy Stats ====="

    print "Final portfolio value: ", port_vals.iloc[-1]
    print "Cumulative return: ", cumulative_return
    print "Std of daily returns: ", std_dr
    print "Mean of daily returns: ", mean_dr

    return port_vals

def _get_portfolio_stats(portfolio_values):
    cumulative_return = (portfolio_values[-1] / portfolio_values[0]) - 1
    daily_returns = (portfolio_values / portfolio_values.shift(1)) - 1
    std_dr = daily_returns.std()
    mean_dr = daily_returns.mean()

    return cumulative_return, std_dr, mean_dr

def _plot(data, long_short_data, title, xlabel, ylabel, colors=['b', 'r', 'g'], legend=None):
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

    # Add vertical green lines indicating LONG entry points
    # Add vertical red lines indicating SHORT entry points.
    for (date, entry_point) in long_short_data:
        color = 'green' if entry_point == 'LONG' else 'red'
        plt.axvline(x=date, color=color, alpha=0.4, linewidth=2.0)

    plt.savefig(
        'report/img/{}.png'.format(title.replace(' ', '')),
        bbox_inches='tight'
    )

if __name__ == '__main__':
    main()
