"""
An improved version of the marketsim code that accepts an orders data frame
"""

__author__ = "Allan Reyes"
__email__ = "reyallan@gatech.edu"
__userId__ = "arx3"

import pandas as pd
from util import get_data

def compute_portvals(orders, start_val=1000000, commission=9.95, impact=0.005, prices=None):
    """
    Computes the daily portfolio values by simulating the execution of trade orders

    Args:
        - orders: A dataframe with the trade orders to simulate
                  It follows the format: Order | Date | Symbol | Shares
                  where Order is either BUY, SELL or HOLD
        - start_val: Starting cash of the portfolio. Defaults to 1,000,000
        - commission: The commission to apply per transaction. Defaults to 9.95
        - impact: The impact of the market in the stocks. Defaults to 0.005
        - prices: An optional DataFrame of prices. If provided, then the
                  data will not be retrieved again. Defaults to None

    Returns:
        - A dataframe with the daily portfolio values
    """

    # Make sure the orders are always sorted
    orders.sort_index(inplace=True)

    start_date, end_date = _extract_start_and_end_dates(orders)
    # Add a column with the dates for easier access than the index
    _add_dates_to_orders(orders)

    symbols = _extract_stock_symbols(orders)

    if prices is not None:
        # Just add the Cash column
        prices['Cash'] = 1.
    else:
        prices = _get_stock_prices_with_cash(symbols, start_date, end_date)

    portvals = _run_market_simulation(prices, orders, start_val, commission, impact)

    return portvals

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

    # Do nothing, i.e. hold the position
    return

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
