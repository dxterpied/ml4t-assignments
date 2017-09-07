"""Unit Tests for optimization.py"""

import datetime as dt
import unittest
import optimization
import numpy as np

class TestOptimization(unittest.TestCase):
    def test_optimize_portfolio_community_1(self):
        params = {
            'start_date': dt.datetime(2008,6,1),
            'end_date': dt.datetime(2009,6,1),
            'symbols': ['IBM', 'X', 'GLD']
        }

        results = {
            'allocations': np.asarray([4.57862602e-01, 3.25260652e-17, 5.42137398e-01]),
            'cumulative_return': -0.0124644030313,
            'average_daily_return': 7.12886342101e-05,
            'volatility': 0.0155967096291,
            'sharpe_ratio': 0.0725583800342
        }

        self._run_test(params, results)

    def test_optimize_portfolio_community_2(self): 
        params = {
            'start_date': dt.datetime(2010,1,1),
            'end_date': dt.datetime(2010,12,31),
            'symbols': ['GOOG', 'AAPL', 'GLD', 'XOM']
        }

        results = {
            'allocations': np.asarray([0.10612268, 0.00777931, 0.54377086, 0.34232715]),
            'cumulative_return': 0.171051073827,
            'average_daily_return': 0.000663584878602,
            'volatility': 0.00828691719462,
            'sharpe_ratio': 1.27117034332
        }

        self._run_test(params, results)

    def test_optimize_portfolio_community_3(self):
        params = {
            'start_date': dt.datetime(2007,1,1),
            'end_date': dt.datetime(2012,12,31),
            'symbols': ['FAKE1', 'FAKE2']
        }

        results = {
            'allocations': np.asarray([0.92971194, 0.07028806]),
            'cumulative_return': 0.950818351627,
            'average_daily_return': 0.00052647684975,
            'volatility': 0.0110333159639,
            'sharpe_ratio': 0.757484052857
        }

        self._run_test(params, results)

    def _run_test(self, params, results):
        start_date = params['start_date']
        end_date = params['end_date']
        symbols = params['symbols']

        allocations, \
        cumulative_return, \
        average_daily_return, \
        volatility, \
        sharpe_ratio = optimization.optimize_portfolio(sd=start_date, ed=end_date, syms=symbols, gen_plot=False)

        self.assertAlmostEqual(np.sum(allocations), 1.0, delta=1e-8, msg="Allocations do not sum to 1.0")

        self.assertTrue(np.allclose(allocations, results['allocations']),
                        msg="Incorrect allocations {}, expected to be {}".format(allocations, results['allocations']))

        self.assertAlmostEqual(cumulative_return, results['cumulative_return'], delta=1e-5,
                               msg="Incorrect cumulative return {}, expected to be {}".format(
                                   cumulative_return, results['cumulative_return']))

        self.assertAlmostEqual(average_daily_return, results['average_daily_return'], delta=1e-5,
                               msg="Incorrect avg. daily return {}, expected to be {}".format(
                                   average_daily_return, results['average_daily_return']))

        self.assertAlmostEqual(volatility, results['volatility'], delta=1e-5,
                               msg="Incorrect volatility {}, expected to be {}".format(
                                   volatility, results['volatility']))

        self.assertAlmostEqual(sharpe_ratio, results['sharpe_ratio'], delta=1e-5,
                               msg="Incorrect sharpe ratio {}, expected to be {}".format(
                                   sharpe_ratio, results['sharpe_ratio']))

if __name__ == '__main__':
    unittest.main(verbosity=2)
