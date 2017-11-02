"""
Implements technical indicators as functions that operate on dataframes

Indicators:
- Momentum
- Bollinger Bands
- Money Flow Index
"""

import datetime as dt
import pandas as pd
from util import get_data

def main():
    """Generates the charts that illustrate the indicators"""
    LOOKBACK = 14
    SYMBOLS = ['JPM']

    date_range = pd.date_range(dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31))
    prices = _get_data(SYMBOLS, date_range, 'Adj Close')
    highs = _get_data(SYMBOLS, date_range, 'High')
    lows = _get_data(SYMBOLS, date_range, 'Low')
    volumes = _get_data(SYMBOLS, date_range, 'Volume')

    dates = prices.index

    mtm = momentum(prices, LOOKBACK)
    sma = simple_moving_average(prices, LOOKBACK)
    bbands = bollinger_bands(prices, sma)
    mfi = money_flow_index(prices, highs, lows, volumes, LOOKBACK)

    _generate_plots(dates, LOOKBACK, prices, mtm, sma, mfi)

def momentum(prices, lookback):
    """
    Computes how much the stock price has changed

    Args:
        - prices: A dataframe with prices
        - lookback: The lookback period

    Returns:
        - A dataframe for the same period with the momentum
    """
    return (prices / prices.shift(lookback)) - 1

def simple_moving_average(prices, lookback):
    """
    Computes the Simple Moving Average for a specific window

    Args:
        - prices: A dataframe with prices
        - lookback: The lookback period

    Returns:
        - A dataframe for the same period with the SMA
    """
    return prices.rolling(window=lookback, min_periods=lookback).mean()

def bollinger_bands(prices, sma):
    """
    Computes the Bollinger Bands using a specified lookback period

    Args:
        - prices: A dataframe with prices
        - sma: The SMA

    Returns:
        - A dataframe for the same period with the Bollinger Bands
    """
    return (prices - sma) / (2. * sma.std())

def money_flow_index(prices, highs, lows, volumes, lookback):
    """
    Computes the Money Flow Index
    http://www.investopedia.com/terms/m/mfi.asp

    Args:
        - prices: A dataframe with prices
        - highs: A dataframe with high prices
        - lows: A dataframe with low prices
        - volumes: A dataframe with share volumes
        - looback: The lookback period

    Returns:
        - A dataframe for the same period with the MFI
    """
    typical_prices = _typical_price(prices, highs, lows)
    raw_money_flows = _raw_money_flow(typical_prices, volumes)
    positive_money_flows = _positive_money_flow(
        raw_money_flows,
        typical_prices,
        lookback
    )
    negative_money_flows = _negative_money_flow(
        raw_money_flows,
        typical_prices,
        lookback
    )
    money_flow_ratio = positive_money_flows / negative_money_flows
    the_money_flow_index = 100 - (100 / (1 + money_flow_ratio))

    return the_money_flow_index

def _typical_price(prices, highs, lows):
    return (prices + highs + lows) / 3.

def _raw_money_flow(typical_prices, volumes):
    return typical_prices * volumes

def _positive_money_flow(raw_money_flows, typical_prices, lookback):
    # Shift typical_prices by 1 to allow comparison with each previous day
    # First day will be Nan, but it makes sure it is not used in the
    # computation of the flow, see: http://tinyurl.com/yq58u6
    return _money_flow(
        raw_money_flows,
        typical_prices,
        lookback,
        typical_prices >= typical_prices.shift(1)
    )

def _negative_money_flow(raw_money_flows, typical_prices, lookback):
    return _money_flow(
        raw_money_flows,
        typical_prices,
        lookback,
        typical_prices < typical_prices.shift(1)
    )

def _money_flow(raw_money_flows, typical_prices, lookback, selector):
    flows = raw_money_flows[selector]
    # Use `fillna(0) and `cumsum()` to obtain the sum of the money
    # flows for each day; `fillna(0)` avoids issues when summing NaN's
    flows = flows.fillna(0).cumsum()
    # And then just retrieve the values for the lookback period
    money_flow = typical_prices.copy()
    # Everything before the lookback period cannot be computed, i.e NaN
    money_flow.iloc[:, :] = float('nan')
    # We need to perform this substraction because the MFI is computed
    # based on differences between days for the lookback period.
    # For example, if the lookback period is 2, and we have a dataframe
    # of 4 total days, then for today, the MFI would be computed based
    # on the difference between today and yesterday, and yesterday and
    # the day before (that is 2-day lookback period); so the values
    # that are used are the money flow for today and the money flow for
    # yesterday. However, if we only use the values for the lookback
    # without removing the ones that are outside of the lookback period,
    # then Pandas would retrieve the values for yesterday and the day
    # before (T-2 days from today) instead of considering today as
    # the first day of the lookback period. For MFI, today is the first
    # day of the lookback period because we look at the difference
    # between successive days, i.e. pairs of days, not actual days
    # before today, like in other indicators, e.g. momentum
    money_flow.values[lookback:, :] = flows.values[lookback:, :] - \
                                      flows.values[:-lookback, :]

    return money_flow

def _get_data(symbols, dates, column):
    data = get_data(symbols, dates, colname=column)
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)
    # Normalize the data
    data = data / data.ix[0, :]

    return data

def _generate_plots(dates, lookback, prices, mtm, sma, mfi):
    # Plot for momentum
    _plot(
        [(dates, prices['JPM']), (dates, mtm['JPM'])],
        'Prices vs {}-day Momentum for JPM'.format(lookback),
        'Date',
        'Normalized Price',
        legend=['Prices', 'Momentum']
    )

    # Plot for Bollinger Bands
    # Bollinger Bands that are going to be plotted
    upper_bband = sma + (2. * sma.std())
    lower_bband = sma + (-2. * sma.std())
    _plot(
        [
            (dates, prices['JPM']),
            (dates, upper_bband['JPM']),
            (dates, lower_bband['JPM']),
            (dates, sma['JPM'])
        ],
        'Prices vs Bollinger Bands vs {}-day SMA for JPM'.format(lookback),
        'Date',
        'Normalized Price',
        colors=['b', 'r', 'r', 'g'],
        legend=['Prices', r'$2\sigma$ upper band', r'$2\sigma$ lower band', 'SMA']
    )

    # Plot for Money Flow Index
    _plot(
        [
            (dates, mfi['JPM'])
        ],
        '{}-day MFI vs MFI limits for JPM'.format(lookback),
        'Date',
        'MFI',
        legend=['MFI'],
        hlines=[(80, '80 - Overbought'), (20, '20 - Oversold')]
    )

def _plot(data, title, xlabel, ylabel, colors=['b', 'r', 'g'], legend=None, hlines=None):
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

    if not hlines is None:
        [plt.axhline(y=hline, color='black', linewidth=2.5, alpha=0.4, label=label)
         for hline, label in hlines]

    if not legend is None:
        plt.legend(fontsize='small')

    plt.savefig(
        'report/img/{}.png'.format(title.replace(' ', '')),
        bbox_inches='tight'
    )

if __name__ == '__main__':
    main()
