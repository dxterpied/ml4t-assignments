"""
Implements a BestPossibleStrategy object
"""

import datetime as dt
import pandas as pd

from marketsimcode import compute_portvals
from util import get_data

class BestPossibleStrategy(object):
    """
    Implements your best possible strategy
    """

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
        prices = BestPossibleStrategy._get_prices(symbol, pd.date_range(sd, ed))
        # Shift the prices one position up to bring tomorrow's price to today's price
        tomorrows_prices = prices.shift(-1)
        # Make sure we are only trading when the market (SPY) traded too!
        # The `prices` dataframe already contains only the trading dates
        trading_dates = prices.index

        return BestPossibleStrategy._generate_orders(symbol, trading_dates, prices, tomorrows_prices)

    @staticmethod
    def _get_prices(symbol, dates):
        prices = get_data([symbol], dates)
        prices.fillna(method='ffill', inplace=True)
        prices.fillna(method='bfill', inplace=True)

        # Only use the symbol we are trading
        # We have to do this because `get_data`
        # add the SPY automatically, and we cannot
        # ask the function to exclude it because
        # we want to make sure we only trade when
        # the market also traded
        prices = prices[symbol]

        return prices

    @staticmethod
    def _generate_orders(symbol, trading_dates, todays_prices, tomorrows_prices):
        orders = pd.DataFrame(index=trading_dates, columns=['Order', 'Date', 'Symbol', 'Shares'])
        current_shares = 0

        for date in trading_dates:
            price_for_tomorrow = tomorrows_prices.loc[date]
            price_for_today = todays_prices.loc[date]

            if BestPossibleStrategy._should_buy(price_for_tomorrow, price_for_today, current_shares):
                order, shares_traded = BestPossibleStrategy._buy(date, symbol, current_shares)
            elif BestPossibleStrategy._should_sell(price_for_tomorrow, price_for_today, current_shares):
                order, shares_traded = BestPossibleStrategy._sell(date, symbol, current_shares)
            else:
                order, shares_traded = BestPossibleStrategy._hold(date, symbol)

            orders.loc[date] = order
            current_shares += shares_traded

        return orders

    @staticmethod
    def _should_buy(price_for_tomorrow, price_for_today, current_shares):
        # Price is going up tomorrow, we should buy
        # unless we already are holding our max long position
        return price_for_tomorrow > price_for_today and current_shares != 1000

    @staticmethod
    def _should_sell(price_for_tomorrow, price_for_today, current_shares):
        # Price is going down tomorrow, we should sell
        # unless we already are holding our max short position
        return price_for_tomorrow < price_for_today and current_shares != -1000

    @staticmethod
    def _buy(date, symbol, current_shares):
        # If we were shorting, we should buy double to reach the positive
        to_buy = 1000
        if current_shares == -1000:
            to_buy = 2000

        return ['BUY', date, symbol, to_buy], to_buy

    @staticmethod
    def _sell(date, symbol, current_shares):
        # If we were longing, we should sell double to reach the negative
        to_sell = 1000
        if current_shares == 1000:
            to_sell = 2000

        return ['SELL', date, symbol, to_sell], to_sell * -1.

    @staticmethod
    def _hold(date, symbol):
        return ['HOLD', date, symbol, 0], 0

def main():
    """
    Generates the charts and metrics for the
    best possible strategy and benchmark
    """
    best_port_vals = _evaluate_best_possible_stategy()
    benchmark_port_vals = _evaluate_benchmark(best_port_vals.index)

    # Normalize values
    best_port_vals = best_port_vals / best_port_vals.iloc[0]
    benchmark_port_vals = benchmark_port_vals / benchmark_port_vals.iloc[0]

    trading_dates = best_port_vals.index

    _plot(
        [
            (trading_dates, benchmark_port_vals),
            (trading_dates, best_port_vals)
        ],
        'Best Possible Strategy vs Benchmark',
        'Date',
        'Normalized Value',
        colors=['blue', 'black'],
        legend=['Benchmark', 'Best Possible Strategy']
    )

def _evaluate_best_possible_stategy():
    bps = BestPossibleStrategy()
    orders = bps.testPolicy(
        symbol="JPM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 12, 31),
        sv=100000
    )
    port_vals = compute_portvals(
        orders,
        start_val=100000,
        commission=0.00,
        impact=0.0
    )
    cumulative_return, std_dr, mean_dr = _get_portfolio_stats(port_vals)

    print "\n===== Best Possible Strategy Stats ====="

    print "Final portfolio value: ", port_vals.iloc[-1]
    print "Cumulative return: ", cumulative_return
    print "Std of daily returns: ", std_dr
    print "Mean of daily returns: ", mean_dr

    return port_vals

def _evaluate_benchmark(trading_dates):
    orders = pd.DataFrame(index=trading_dates, columns=['Order', 'Date', 'Symbol', 'Shares'])
    # Buy and hold position
    orders.iloc[0] = ['BUY', trading_dates[0], 'JPM', 1000]
    orders.iloc[1:] = [['HOLD', date, 'JPM', 0] for date in trading_dates[1:]]

    port_vals = compute_portvals(
        orders,
        start_val=100000,
        commission=0.00,
        impact=0.0
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

def _plot(data, title, xlabel, ylabel, colors=['b', 'r', 'g'], legend=None):
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

    plt.savefig(
        'report/img/{}.png'.format(title.replace(' ', '')),
        bbox_inches='tight'
    )

if __name__ == '__main__':
    main()
