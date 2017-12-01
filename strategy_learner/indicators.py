"""
Implements technical indicators as functions that operate on dataframes

Indicators:
- Momentum
- Price/SMA ratio
- Bollinger Bands
- Money Flow Index
"""

__author__ = "Allan Reyes"
__email__ = "reyallan@gatech.edu"
__userId__ = "arx3"

import datetime as dt
import pandas as pd
from util import get_data

def main():
    """Generates the charts that illustrate the indicators"""
    LOOKBACK = 10
    SYMBOLS = ['JPM']

    date_range = pd.date_range(dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31))
    prices = _get_data(SYMBOLS, date_range, 'Adj Close')
    highs = _get_data(SYMBOLS, date_range, 'High')
    lows = _get_data(SYMBOLS, date_range, 'Low')
    volumes = _get_data(SYMBOLS, date_range, 'Volume')

    dates = prices.index

    mtm = momentum(prices, LOOKBACK)
    sma, sma_ratio = simple_moving_average(prices, LOOKBACK)
    bbands = bollinger_bands(prices, sma, LOOKBACK)
    mfi = money_flow_index(prices, highs, lows, volumes, LOOKBACK)

    _save_indicators(
        (mtm, sma_ratio, bbands, mfi),
        ['Momentum', 'SMA', 'BollingerBands', 'MFI']
    )

    _generate_plots(dates, LOOKBACK, prices, mtm, sma, sma_ratio, bbands, mfi)

def momentum(prices, lookback):
    """
    Computes how much the stock price has changed

    Args:
        - prices: A dataframe with prices
        - lookback: The lookback period

    Returns:
        - A dataframe for the same period with the standarized momentum
    """
    return _standarize_indicator((prices / prices.shift(lookback)) - 1)

def simple_moving_average(prices, lookback):
    """
    Computes the Simple Moving Average for a specific window
    and the Price/SMA ratio which is used for technical analysis

    Args:
        - prices: A dataframe with prices
        - lookback: The lookback period

    Returns:
        - A dataframe for the same period with the SMA
        - A dataframe for the same period with the Price/SMA standarized ratio
    """
    sma = prices.rolling(window=lookback, min_periods=lookback).mean()
    return sma, _standarize_indicator((prices / sma) - 1)

def bollinger_bands(prices, sma, lookback):
    """
    Computes the Bollinger Bands using a specified lookback period

    Args:
        - prices: A dataframe with prices
        - sma: The SMA

    Returns:
        - A dataframe for the same period with the standarized Bollinger Bands
    """
    # Bolling Bands use the rolling standard deviation over the same lookback
    # period as the rolling mean
    std = prices.rolling(window=lookback, min_periods=lookback).std()
    return _standarize_indicator((prices - sma) / (2. * std))

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

    return data

def _standarize_indicator(indicator):
    return (indicator - indicator.mean()) / indicator.std()

def _save_indicators(indicators, names):
    for index, indicator in enumerate(indicators):
        indicator.to_csv("report/indicators/{}.csv".format(names[index]))

def _generate_plots(dates, lookback, prices, mtm, sma, sma_ratio, bbands, mfi):
    # Plot the prices for JPM
    _plot(
        {
            'data' : [(dates, prices['JPM'])],
            'color' : ['blue'],
            'legend' : ['Prices'],
            'title' : 'Prices for JPM',
            'xlabel' : 'Date',
            'ylabel' : 'Price'
        },
        filename='PricesForJPM'
    )

    # Plot the Price/SMA ratio
    _plot(
        {
            'data' : [(dates, sma_ratio['JPM'])],
            'color' : ['red'],
            'legend' : ['Price/SMA Ratio'],
            'title' : 'Price/SMA Ratio for JPM',
            'xlabel' : 'Date',
            'ylabel' : 'Standarized Price/SMA Ratio'
        },
        filename='PriceToSMARatioForJPM'
    )

    # Plot the Momentum for JPM
    _plot(
        {
            'data' : [(dates, mtm['JPM'])],
            'color' : ['red'],
            'legend' : ['Momentum'],
            'title' : '{}-day Momentum for JPM'.format(lookback),
            'xlabel' : 'Date',
            'ylabel' : 'Standarized Momentum'
        },
        filename='{}-dayMomentumForJPM'.format(lookback)
    )

    # Plot the Bollinger Bands against Prices and SMA for JPM
    # Bollinger Bands that are going to be plotted
    std = prices.rolling(window=lookback, min_periods=lookback).std()
    upper_bband = sma + (2. * std)
    lower_bband = sma + (-2. * std)
    _plot(
        {
            'data' : [
                (dates, prices['JPM']),
                (dates, upper_bband['JPM']),
                (dates, lower_bband['JPM']),
                (dates, sma['JPM'])
            ],
            'color' : ['blue', 'red', 'red', 'green'],
            'legend' : [
                'Prices',
                r'$2\sigma$ upper band',
                r'$2\sigma$ lower band',
                'SMA'
            ],
            'title' : 'Prices vs Bollinger Bands vs {}-day SMA for JPM'.format(lookback),
            'xlabel' : 'Date',
            'ylabel' : 'Price'
        },
        filename='PricesvsBollingerBandsvs{}-daySMAforJPM'.format(lookback)
    )

    # Plot Bollinger Band values
    _plot(
        {
            'data' : [(dates, bbands['JPM'])],
            'color' : ['red'],
            'legend' : ['Bollinger Band Values'],
            'title' : 'Bollinger Band Values for JPM',
            'xlabel' : 'Date',
            'ylabel' : 'Standarized Bollinger Band Values',
            'hlines': [(1, '1 - SELL signal'), (-1, '-1 - BUY signal')]
        },
        filename='BollingerBandsValuesforJPM'
    )

    # Plot for Money Flow Index
    _plot(
        {
            'data' : [(dates, mfi['JPM'])],
            'color' : ['red'],
            'legend' : ['MFI'],
            'title' : '{}-day MFI for JPM'.format(lookback),
            'xlabel' : 'Date',
            'ylabel' : 'MFI',
            'hlines': [(70, '70 - Overbought'), (30, '30 - Oversold')]
        },
        filename='{}-dayMFIforJPM'.format(lookback)
    )

def _plot(plot, filename='plot.png'):
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

    data = plot['data']
    color = plot['color']
    legend = plot['legend']
    title = plot['title']
    xlabel = plot['xlabel']
    ylabel = plot['ylabel']
    # hlines is optional so it can be None
    # `get` returns None instead of throwing an error
    hlines = plot.get('hlines')

    for index, (xdata, ydata) in enumerate(data):
        plt.plot(
            xdata,
            ydata,
            linewidth=2.5,
            color=color[index],
            alpha=0.4,
            label=legend[index]
        )

        plt.title(title)
        plt.xlabel(xlabel, fontsize='15')
        plt.ylabel(ylabel, fontsize='15')

        if not hlines is None:
            [plt.axhline(y=hline, color='black', linewidth=2.5, alpha=0.4, label=label)
             for hline, label in hlines]

        plt.legend(fontsize='small')

    plt.savefig(
        'report/img/{}.png'.format(filename),
        bbox_inches='tight'
    )

if __name__ == '__main__':
    main()
