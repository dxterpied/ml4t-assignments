"""Unit tests for StrategyLearner.py"""

__author__ = "Allan Reyes"
__email__ = "reyallan@gatech.edu"
__userId__ = "arx3"

import datetime as dt
import numpy as np
import pandas as pd
import unittest

from marketsimcode import compute_portvals
from StrategyLearner import StrategyLearner
from util import get_data

class StrategyLearnerTest(unittest.TestCase):
    def test_in_sample_for_unknown_symbol(self):

        start_date = dt.datetime(2008,1,1)
        end_date = dt.datetime(2009,12,31)
        symbol = 'AMZN'
        start_value = 100000
        impact = 0.0

        benchmark = self._compute_benchmark(
            start_date,
            end_date,
            start_value,
            symbol,
            impact
        )

        sl = StrategyLearner(impact=impact)

        sl.addEvidence(symbol=symbol, sd=start_date, ed=end_date, sv=start_value)
        trades = sl.testPolicy(symbol=symbol, sd=start_date, ed=end_date, sv=start_value)

        orders = self._generate_orders(symbol, trades)
        cumulative_return = self._compute_performance(orders, start_value, impact)

        self.assertTrue(cumulative_return > benchmark,
                        msg="Learner did not beat benchmark: {} to {}".format(
                            cumulative_return, benchmark))

    def test_in_sample_to_verify_market_impact(self):
        start_date = dt.datetime(2008,1,1)
        end_date = dt.datetime(2009,12,31)
        symbol = 'AMZN'
        start_value = 100000
        impact_1 = 0.0
        impact_2 = 5.0

        sl_1 = StrategyLearner(impact=impact_1)
        sl_1.addEvidence(symbol=symbol, sd=start_date, ed=end_date, sv=start_value)
        trades_1 = sl_1.testPolicy(symbol=symbol, sd=start_date, ed=end_date, sv=start_value)

        sl_2 = StrategyLearner(impact=impact_2)
        sl_2.addEvidence(symbol=symbol, sd=start_date, ed=end_date, sv=start_value)
        trades_2 = sl_2.testPolicy(symbol=symbol, sd=start_date, ed=end_date, sv=start_value)

        percent_different = float(np.sum((trades_1 != trades_2).values)) / float(trades_1.shape[0])

        self.assertTrue(percent_different > 0.10,
                        msg="Learner did not return more than 10% different trades")

    def _compute_benchmark(self, start_date, end_date, start_value, symbol, market_impact):
        trading_dates = get_data([symbol,], pd.date_range(start_date, end_date)).index
        orders = pd.DataFrame(index=trading_dates, columns=['Order', 'Date', 'Symbol', 'Shares'])

        # Buy 1000 shares on the first trading day, sell 1000 shares on the last day
        orders.iloc[0] = ['BUY', trading_dates[0], symbol, 1000]
        orders.iloc[1:] = [['HOLD', date, symbol, 0] for date in trading_dates[1:]]

        return self._compute_performance(orders, start_value, market_impact)

    def _compute_performance(self, orders, start_value, impact):
        portfolio_values = compute_portvals(orders, start_val=start_value, commission=0.0, impact=impact)
        return (portfolio_values[-1] / portfolio_values[0]) - 1

    def _generate_orders(self, symbol, trades):
        # Generate an orders DataFrame as required by my marketsimcode
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

if __name__ == '__main__':
    unittest.main(verbosity=2)