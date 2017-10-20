"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000, commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    # TODO: Your code here

    orders = _scan_orders(orders_file)

    start_date, end_date = _extract_start_and_end_dates(orders)

    # Add a column with the dates for easier access than the index
    _add_dates_to_orders(orders)

    symbols = _extract_stock_symbols(orders)

    prices = _get_stock_prices_with_cash(symbols, start_date, end_date)

    portvals = _run_market_simulation(prices, orders, start_val, commission, impact)

    return portvals

def author():
    return 'arx3'

def _scan_orders(orders_file):
    orders = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'], keep_date_col=True)
    # Make sure the orders are sorted by Date, i.e. the index
    orders.sort_index(inplace=True)

    return orders

def _extract_start_and_end_dates(orders):
    assert isinstance(orders, pd.DataFrame)
    # The Date column is converted into an index, so the only
    # way to extract the first and last dates is by using the
    # index property
    return orders.head(1).index[0], orders.tail(1).index[0]

def _add_dates_to_orders(orders):
    orders['Date'] = orders.index

def _extract_stock_symbols(orders):
    assert isinstance(orders, pd.DataFrame)
    return list(orders['Symbol'].unique())

def _get_stock_prices_with_cash(symbols, start_date, end_date):
    assert isinstance(symbols, list)

    prices = get_data(symbols, pd.date_range(start_date, end_date))

    # Always fill-forward first!
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)

    # Only use the stocks we have in our portfolio
    prices = prices[symbols]

    # Add a Cash column with a constant value of $1.00
    # This is the price of a "share" of USD
    prices['Cash'] = 1.

    return prices

def _run_market_simulation(prices, orders, start_val, commission, impact):
    # The `trades` DataFrame will track *changes* in shares and cash after each order
    trades = prices.copy()
    trades[:] = 0

    # The `holdings` DataFrame will represent the amount of shares and cash we have
    holdings = prices.copy()
    # When we entered the market, we didn't have any stocks
    # and our cash constitued our initial funds
    holdings[:] = 0
    holdings.iloc[0]['Cash'] = start_val

    # The `values` DataFrame will hold the values of the portfolio for each day
    values = prices.copy()

    for _, order in orders.iterrows():
        _execute_order(order, prices, trades, commission, impact)

    _update_holdings(holdings, trades)

    _update_values(values, holdings, prices)

    # Compute the total value of the portfolio as the sum of equities and cash
    # This means a sum over columns for each row, hence axis=1
    values = values.sum(axis=1)

    return values

def _execute_order(order, prices, trades, commission, impact):
    order_type = order['Order']

    if order_type == 'BUY':
        _execute_buy_order(order, prices, trades, commission, impact)
    elif order_type == 'SELL':
        _execute_sell_order(order, prices, trades, commission, impact)
    else:
        raise RuntimeError("Invalid order type: {}".format(order_type))

def _execute_buy_order(order, prices, trades, commission, impact):
    date = order['Date']
    symbol = order['Symbol']
    shares = order['Shares']

    trades.ix[date, symbol] += shares
    # `impact` makes the price increase before the buy, i.e. I pay more money than expected
    trades.ix[date, 'Cash'] += shares * (prices.ix[date, symbol] * (1. + impact)) * -1. - commission

def _execute_sell_order(order, prices, trades, commision, impact):
    date = order['Date']
    symbol = order['Symbol']
    shares = order['Shares']

    trades.ix[date, symbol] += shares * -1.
    # `impact` makes the price drop before the sell, i.e. I make less money than expected
    trades.ix[date, 'Cash'] += shares * (prices.ix[date, symbol] * (1. - impact)) - commision

def _update_holdings(holdings, trades):
    index = 0
    for _, trade in trades.iterrows():
        if index == 0:
            # No previous data so just use the current trade
            holdings.iloc[index] += trade
        else:
            # The previous holdings plus the current trade
            holdings.iloc[index] = holdings.iloc[index - 1] + trade

        index += 1

def _update_values(values, holdings, prices):
    values[:] = prices * holdings

def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders-02.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2008,6,1)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2,0.01,0.02,1.5]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

if __name__ == "__main__":
    test_code()
