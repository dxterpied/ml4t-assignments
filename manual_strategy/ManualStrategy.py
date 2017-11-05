"""
Implements a ManualStrategy object
"""

import datetime as dt
import pandas as pd

from indicators import bollinger_bands, momentum, money_flow_index, simple_moving_average
from marketsimcode import compute_portvals
from util import get_data

class ManualStrategy(object):
    """
    Implements your manual strategy
    """

    def __init__(self):
        # Keeps track of our LONG and SHORT entry points
        self._long_short_entries = []

    def get_long_short_entries(self):
        return self._long_short_entries

    def testPolicy(self, symbol="AAPL", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv=100000):
        """
        Computes the best possible trading strategy

        Args:
            - symbol: The stock symbol to use
            - sd: The start date
            - ed: The end date
            - sv: The starting value of the portfolio

        Returns:
            - A dataframe of orders of the form: Order | Date | Symbol | Shares
        """
        date_range = pd.date_range(sd, ed)
        prices, \
        highs, \
        lows, \
        volumes = ManualStrategy._get_prices_and_volume(symbol, date_range)

        mtm, \
        sma, \
        bbands, \
        mfi = ManualStrategy._compute_indicators(prices, highs, lows, volumes, 10)

        trading_dates = prices.index

        return self._generate_orders(symbol, trading_dates, (mtm, sma, bbands, mfi))

    @staticmethod
    def _get_prices_and_volume(symbol, date_range):
        return ManualStrategy._get_data(symbol, date_range, 'Adj Close'), \
               ManualStrategy._get_data(symbol, date_range, 'High'), \
               ManualStrategy._get_data(symbol, date_range, 'Low'), \
               ManualStrategy._get_data(symbol, date_range, 'Volume')

    @staticmethod
    def _get_data(symbol, dates, column):
        data = get_data([symbol], dates, colname=column)
        data.fillna(method='ffill', inplace=True)
        data.fillna(method='bfill', inplace=True)

        return data

    @staticmethod
    def _compute_indicators(prices, highs, lows, volumes, lookback):
        mtm = momentum(prices, lookback)
        sma, sma_ratio = simple_moving_average(prices, lookback)
        bbands = bollinger_bands(prices, sma)
        mfi = money_flow_index(prices, highs, lows, volumes, lookback)

        return mtm, sma_ratio, bbands, mfi

    def _generate_orders(self, symbol, trading_dates, indicators):
        orders = pd.DataFrame(index=trading_dates, columns=['Order', 'Date', 'Symbol', 'Shares'])
        current_shares = 0

        for index, date in enumerate(trading_dates):
            # Use the current date (i.e. today) as yesterday for the first trading day
            yesterday = trading_dates[index - 1] if index > 0 else date

            if ManualStrategy._should_long(symbol, date, yesterday, indicators, current_shares):
                order, shares_traded = ManualStrategy._long(date, symbol, current_shares)
                self._long_short_entries.append((date, 'LONG'))
            elif ManualStrategy._should_short(symbol, date, yesterday, indicators, current_shares):
                order, shares_traded = ManualStrategy._short(date, symbol, current_shares)
                self._long_short_entries.append((date, 'SHORT'))
            elif ManualStrategy._should_close_long(symbol, date, yesterday, indicators, current_shares):
                order, shares_traded = ManualStrategy._sell(date, symbol)
            elif ManualStrategy._should_close_short(symbol, date, yesterday, indicators, current_shares):
                order, shares_traded = ManualStrategy._buy(date, symbol)
            else:
                order, shares_traded = ManualStrategy._hold(date, symbol)

            orders.loc[date] = order
            current_shares += shares_traded

        return orders

    @staticmethod
    def _should_long(symbol, today, yesterday, indicators, current_shares):
        # Look for a long position if the stock seems to be oversold
        # and we are currently not at our max long position (1000 shares)
        # We define oversold as:
        # MFI < 30 -> by definition (http://www.investopedia.com/terms/m/mfi.asp)
        # Bollinger Bands crossing from below and up, i.e. the difference
        # between the price and the SMA is negatively larger than 2 * std but
        # it is trending up
        _, _, bbands, mfi = indicators

        mfi_oversold = mfi.loc[today, symbol] < 30
        bband_crossing_below_and_up = bbands.loc[today, symbol] < -1.

        return mfi_oversold and \
               bband_crossing_below_and_up and \
               current_shares != 1000

    @staticmethod
    def _should_short(symbol, today, yesterday, indicators, current_shares):
        # Look for a short position if the stock seems to be overbought
        # and we are currently not at our max short position (-1000 shares)
        # We define overbought as:
        # 1) MFI > 70 -> by definition (http://www.investopedia.com/terms/m/mfi.asp)
        # 2) Bollinger Bands crossing from above and down, i.e. the difference
        #    between the price and the SMA is positively larger than 2 * std but
        #    it is trending down
        _, _, bbands, mfi = indicators

        mfi_overbought = mfi.loc[today, symbol] > 70
        bband_crossing_above_and_down = bbands.loc[today, symbol] > 1

        return mfi_overbought and \
               bband_crossing_above_and_down and \
               current_shares != -1000

    @staticmethod
    def _should_close_long(symbol, today, yesterday, indicators, current_shares):
        # Look for a trend where the price is coming down and we are currently
        # holding a long position (1000 shares). In this case, we don't want to
        # move to a short position immediately since there is no strong indicative
        # of that. We rather want to just close the position to avoid losing money
        # We identify this trend by either:
        # a) Yesterday, the price/SMA ratio was above 1 and today, it is less than or exactly 1
        #    which indicates that the price is regressing down to the SMA
        # b) There is a drop in momentum from yesterday to today of more than 50%
        #    which reinforces the drop identified by the price/SMA ratio
        mtm, sma, _, _ = indicators

        sma_trending_down = sma.loc[yesterday, symbol] > 1 and sma.loc[today, symbol] <= 1

        mtm_today = mtm.loc[today, symbol]
        mtm_yesterday = mtm.loc[yesterday, symbol]
        drop_in_momentum = mtm_today < mtm_yesterday and \
                           (mtm_yesterday - mtm_today) / mtm_yesterday > 0.50

        return (sma_trending_down or \
               drop_in_momentum) and \
               current_shares == 1000

    @staticmethod
    def _should_close_short(symbol, today, yesterday, indicators, current_shares):
        # Look for a trend where the price is coming up and we are currently
        # holding a short position (-1000 shares). In this case, we don't want to
        # move to a long position immediately since there is no strong indicative
        # of that. We rather want to just close the position to avoid losing money
        # We identify this trend by:
        # a) Yesterday, the price/SMA ratio was less than 1 and today, it is above or exactly 1
        #    which indicates that the price is regressing up to the SMA
        # b) There is an increase in momentum from yesterday to today of more than 50%
        #    which reinforces the increase identified by the price/SMA ratio
        mtm, sma, _, _ = indicators

        sma_trending_up = sma.loc[yesterday, symbol] < 1 and sma.loc[today, symbol] >= 1

        mtm_today = mtm.loc[today, symbol]
        mtm_yesterday = mtm.loc[yesterday, symbol]
        increase_in_momentum = mtm_today > mtm_yesterday and \
                               (mtm_today - mtm_yesterday) / mtm_yesterday > 0.50

        return (sma_trending_up or \
               increase_in_momentum) and \
               current_shares == -1000

    @staticmethod
    def _long(date, symbol, current_shares):
        to_buy = 1000
        # If we were shorting, buy double to reach the long position
        if current_shares == -1000:
            to_buy = 2000

        return ['BUY', date, symbol, to_buy], to_buy

    @staticmethod
    def _short(date, symbol, current_shares):
        to_sell = 1000
        # If we were longing, sell double to reach the short position
        if current_shares == 1000:
            to_sell = 2000

        return ['SELL', date, symbol, to_sell], to_sell * -1.

    @staticmethod
    def _buy(date, symbol):
        return ['BUY', date, symbol, 1000], 1000

    @staticmethod
    def _sell(date, symbol):
        return ['SELL', date, symbol, 1000], -1000

    @staticmethod
    def _hold(date, symbol):
        return ['HOLD', date, symbol, 0], 0

def main():
    """
    Generates the charts and metrics for the
    manual strategy and benchmark
    """
    manual_port_vals, manual_orders, long_short_data = _evaluate_manual_stategy()
    benchmark_port_vals = _evaluate_benchmark(manual_port_vals.index)

    # Save the orders generated for debugging purposes
    manual_orders.to_csv('report/notes/manual_strategy_orders_in_sample.csv')

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
        'Manual Strategy vs Benchmark - In Sample',
        'Date',
        'Normalized Value',
        colors=['blue', 'black'],
        legend=['Benchmark', 'Manual Strategy']
    )

def _evaluate_manual_stategy():
    ms = ManualStrategy()
    orders = ms.testPolicy(
        symbol="JPM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 12, 31),
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
