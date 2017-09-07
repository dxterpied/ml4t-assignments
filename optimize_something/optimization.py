"""MC1-P2: Optimize a portfolio."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data
import scipy.optimize as spo

# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY

    _fill_missing_prices(prices_all)

    prices_all = _normalize(prices_all)

    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # Find the allocations for the optimal portfolio
    allocs = _find_optimal_allocations(prices)

    # Get daily portfolio value
    port_val = _compute_daily_portfolio_values(prices, allocs, 1000000)

    # Get portfolio statistics
    cr, adr, sddr, sr = _aggregate_portfolio_stats(port_val, 0.0, 252)

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        port_val = _normalize(port_val)
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        _save_plot(df_temp)

    return allocs, cr, adr, sddr, sr

def _fill_missing_prices(prices):
    # Fill forward
    prices.fillna(method='pad', inplace=True)
    # Fill backward
    prices.fillna(method='bfill', inplace=True)

def _normalize(values):
    return values / values.ix[0, :]

def _find_optimal_allocations(prices):
    number_of_stocks = prices.shape[1]

    initial_allocations = [1. / number_of_stocks for n in range(number_of_stocks)]

    # Minimize volatility, i.e. standard deviation of daily returns
    bounds = [(0.0, 1.0) for n in range(number_of_stocks)]
    constraints = {
        'type': 'eq',
        # All allocations must sum to 1.0 which means this function
        # will return a 0 when the constraint is satisfied
        'fun': lambda allocations: np.sum(allocations) - 1.0
    }
    result = spo.minimize(
        _compute_volatity_for_minimizer,
        initial_allocations,
        args=(prices),
        method='SLSQP',
        bounds=bounds,
        constraints=(constraints)
    )

    return result.x

def _compute_volatity_for_minimizer(allocations, prices):
    portfolio_value = _compute_daily_portfolio_values(prices, allocations, 1000000)

    return _compute_standard_deviation_daily_returns(portfolio_value)

def _compute_daily_portfolio_values(prices, allocations, starting_value):
    assert isinstance(prices, pd.DataFrame)
    return (prices.multiply(allocations) * starting_value).sum(axis=1)

def _aggregate_portfolio_stats(portfolio_value, risk_free_rate, sample_frequency):
    return _compute_cumulative_return(portfolio_value), \
           _compute_average_daily_returns(portfolio_value), \
           _compute_standard_deviation_daily_returns(portfolio_value), \
           _compute_shape_ratio(portfolio_value, risk_free_rate, sample_frequency)

def _compute_cumulative_return(portfolio_value):
    return (portfolio_value[-1] / portfolio_value[0]) - 1

def _compute_average_daily_returns(portfolio_value):
    return _compute_daily_returns(portfolio_value).mean()

def _compute_standard_deviation_daily_returns(portfolio_value):
    return _compute_daily_returns(portfolio_value).std()

def _compute_shape_ratio(portfolio_value, risk_free_rate, sample_frequency):
    daily_returns = _compute_daily_returns(portfolio_value)

    return np.sqrt(sample_frequency) * ((daily_returns - risk_free_rate).mean() / daily_returns.std())

def _compute_daily_returns(portfolio_value):
    return (portfolio_value / portfolio_value.shift(1)) - 1

def _save_plot(df):
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter

    ax = df.plot(title="Daily Portfolio Value and SPY", fontsize=12)
    ax.grid(color='black', linestyle='dotted')
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized price")
    ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))

    plt.savefig("./plot.pdf", bbox_inches='tight')

def generate_report_chart():
    """Generates the chart for the report"""
    start_date = dt.datetime(2008,6,1)
    end_date = dt.datetime(2009,6,1)
    symbols = ['IBM', 'X', 'GLD']

    allocations, \
    cumulative_return, \
    average_daily_return, \
    volatility, \
    sharpe_ratio = optimize_portfolio(sd=start_date, ed=end_date, syms=symbols, gen_plot=True)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sharpe_ratio
    print "Volatility (stdev of daily returns):", volatility
    print "Average Daily Return:", average_daily_return
    print "Cumulative Return:", cumulative_return

if __name__ == "__main__":
    generate_report_chart()
