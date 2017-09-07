"""Analyze a portfolio."""

import pandas as pd
import numpy as np
import datetime as dt
from util import get_data, plot_data

# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def assess_portfolio(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,1,1), \
    syms = ['GOOG','AAPL','GLD','XOM'], \
    allocs=[0.1,0.2,0.3,0.4], \
    sv=1000000, rfr=0.0, sf=252.0, \
    gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY

    _fill_missing_prices(prices_all)

    prices_all = _normalize(prices_all)

    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # Get daily portfolio value
    port_val = _compute_daily_portfolio_values(prices, allocs, sv)

    # Get portfolio statistics (note: std_daily_ret = volatility)
    cr, adr, sddr, sr = _aggregate_portfolio_stats(port_val, rfr, sf)

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        port_val = _normalize(port_val)
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        _save_plot(df_temp)

    ev = port_val[-1]

    return cr, adr, sddr, sr, ev

def _fill_missing_prices(prices):
    # Fill forward
    prices.fillna(method='pad', inplace=True)
    # Fill backward
    prices.fillna(method='bfill', inplace=True)

def _normalize(values):
    return values / values.ix[0, :]

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
    ax = df.plot(title="Daily portfolio value and SPY", fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized price")
    plt.savefig("./plot.png", bbox_inches='tight')

def test_code():
    # This code WILL NOT be tested by the auto grader
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!
    start_date = dt.datetime(2009,1,1)
    end_date = dt.datetime(2010,1,1)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    allocations = [0.2, 0.3, 0.4, 0.1]
    start_val = 1000000  
    risk_free_rate = 0.0
    sample_freq = 252

    # Assess the portfolio
    cr, adr, sddr, sr, ev = assess_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        allocs = allocations,\
        sv = start_val, \
        gen_plot = False)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr

if __name__ == "__main__":
    test_code()
