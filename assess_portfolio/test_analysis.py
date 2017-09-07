"""Unit Tests for analysis.py"""

import datetime as dt
import unittest
import analysis

class TestAnalysis(unittest.TestCase):
    def test_assess_portfolio_example_1(self):
        params = {
            'start_date': dt.datetime(2010,1,1),
            'end_date': dt.datetime(2010,12,31),
            'symbols': ['GOOG', 'AAPL', 'GLD', 'XOM'],
            'allocations': [0.2, 0.3, 0.4, 0.1],
            'start_value': 1000000,
            'risk_free_rate': 0.0,
            'sample_freq': 252
        }

        results = {
            'cumulative_return': 0.255646784534,
            'average_daily_return': 0.000957366234238,
            'volatility': 0.0100104028,
            'sharpe_ratio': 1.51819243641,
            'end_value': 1255646.78453
        }

        self._run_test(params, results)

    def test_assess_portfolio_example_2(self):
        params = {
            'start_date': dt.datetime(2010,1,1),
            'end_date': dt.datetime(2010,12,31),
            'symbols': ['AXP', 'HPQ', 'IBM', 'HNZ'],
            'allocations': [0.0, 0.0, 0.0, 1.0],
            'start_value': 1000000,
            'risk_free_rate': 0.0,
            'sample_freq': 252
        }

        results = {
            'cumulative_return': 0.198105963655,
            'average_daily_return': 0.000763106152672,
            'volatility': 0.00926153128768,
            'sharpe_ratio': 1.30798398744,
            'end_value': 1198105.96365
        }

        self._run_test(params, results)

    def test_assess_portfolio_example_3(self):
        params = {
            'start_date': dt.datetime(2010,6,1),
            'end_date': dt.datetime(2010,12,31),
            'symbols': ['GOOG', 'AAPL', 'GLD', 'XOM'],
            'allocations': [0.2, 0.3, 0.4, 0.1],
            'start_value': 1000000,
            'risk_free_rate': 0.0,
            'sample_freq': 252
        }

        results = {
            'cumulative_return': 0.205113938792,
            'average_daily_return': 0.00129586924366,
            'volatility': 0.00929734619707,
            'sharpe_ratio': 2.21259766672,
            'end_value': 1205113.93879
        }

        self._run_test(params, results)

    def test_assess_portfolio_missing_data(self):
        params = {
            'start_date': dt.datetime(2007,1,1),
            'end_date': dt.datetime(2010,12,31),
            'symbols': ['FAKE1', 'FAKE2'],
            'allocations': [0.5, 0.5],
            'start_value': 1000000,
            'risk_free_rate': 0.0,
            'sample_freq': 252
        }

        results = {
            'cumulative_return': -0.0711700719903,
            'average_daily_return': 9.35473706744e-06,
            'volatility': 0.0128072822706,
            'sharpe_ratio': 0.0115951100341,
            'end_value': None
        }

        self._run_test(params, results)

    def test_assess_portfolio_community_1(self):
        params = {
            'start_date': dt.datetime(2010,9,1),
            'end_date': dt.datetime(2010,12,31),
            'symbols': ['IYR'],
            'allocations': [1.0],
            'start_value': 1000000,
            'risk_free_rate': 0.01,
            'sample_freq': 252
        }

        results = {
            'cumulative_return': 0.0849230769231,
            'average_daily_return': 0.00102951730912,
            'volatility': 0.0108856018686,
            'sharpe_ratio': -13.0816834715,
            'end_value': 1084923.07692
        }

        self._run_test(params, results)

    def test_assess_portfolio_community_2(self):
        params = {
            'start_date': dt.datetime(2009,7,2),
            'end_date': dt.datetime(2010,7,30),
            'symbols': ['USB','VAR'],
            'allocations': [0.3, 0.7],
            'start_value': 1000000,
            'risk_free_rate': 0.02,
            'sample_freq': 252
        }

        results = {
            'cumulative_return': 0.577665964655,
            'average_daily_return': 0.00179946813815,
            'volatility': 0.0152327985726,
            'sharpe_ratio': -18.9672623081,
            'end_value': 1577665.96466
        }

        self._run_test(params, results)

    def test_assess_portfolio_community_3(self):
        params = {
            'start_date': dt.datetime(2008,6,3),
            'end_date': dt.datetime(2010,6,29),
            'symbols': ['HSY','VLO','HOT'],
            'allocations': [0.2, 0.4, 0.4],
            'start_value': 1000000,
            'risk_free_rate': 0.03,
            'sample_freq': 252
        }

        results = {
            'cumulative_return': -0.202916389755,
            'average_daily_return': -4.43318599737e-06,
            'volatility': 0.0292933210871,
            'sharpe_ratio': -16.2598706108,
            'end_value': 797083.610245
        }

        self._run_test(params, results)

    def test_assess_portfolio_community_4(self):
        params = {
            'start_date': dt.datetime(2007,5,4),
            'end_date': dt.datetime(2010,5,28),
            'symbols': ['VNO','WU','EMC','AMGN'],
            'allocations': [0.2, 0.3, 0.4, 0.1],
            'start_value': 1000000,
            'risk_free_rate': 0.04,
            'sample_freq': 252
        }

        results = {
            'cumulative_return': -0.0602831013866,
            'average_daily_return': 0.000183438771035,
            'volatility': 0.0230227053156,
            'sharpe_ratio': -27.4541286863,
            'end_value': 939716.898613
        }

        self._run_test(params, results)

    def test_assess_portfolio_community_5(self):
        params = {
            'start_date': dt.datetime(2006,4,5),
            'end_date': dt.datetime(2010,4,26),
            'symbols': ['ADSK','BXP','IGT','SWY','PH'],
            'allocations': [0.2, 0.3, 0.1, 0.2, 0.2],
            'start_value': 1000000,
            'risk_free_rate': 0.05,
            'sample_freq': 252
        }

        results = {
            'cumulative_return': 0.0766844908523,
            'average_daily_return': 0.000315946476717,
            'volatility': 0.0220751167394,
            'sharpe_ratio': -35.7284587801,
            'end_value': 1076684.49085
        }

        self._run_test(params, results)

    def test_assess_portfolio_community_6(self):
        params = {
            'start_date': dt.datetime(2005,4,6),
            'end_date': dt.datetime(2010,3,25),
            'symbols': ['ETN','KSS','NYT','GPS','BMC','TEL'],
            'allocations': [0.2, 0.1, 0.1, 0.1, 0.4, 0.1],
            'start_value': 1000000,
            'risk_free_rate': 0.06,
            'sample_freq': 252
        }

        results = {
            'cumulative_return': 0.626452592439,
            'average_daily_return': 0.000534742431109,
            'volatility': 0.0170978584343,
            'sharpe_ratio': -55.2105225742,
            'end_value': 1626452.59244
        }

        self._run_test(params, results)

    def test_assess_portfolio_community_7(self):
        params = {
            'start_date': dt.datetime(2003,2,8),
            'end_date': dt.datetime(2010,1,23),
            'symbols': ['HRL','SDS','ACS','IFF','WMB','FFIV','BK','AIV'],
            'allocations': [0.2, 0.2, 0.1, 0.1, 0.2, 0, 0.2, 0],
            'start_value': 1000000,
            'risk_free_rate': 0.08,
            'sample_freq': 252
        }

        results = {
            'cumulative_return': 1.73594958129,
            'average_daily_return': 0.000708444528675,
            'volatility': 0.0162885689486,
            'sharpe_ratio': -77.2759365809,
            'end_value': 2735949.58129
        }

        self._run_test(params, results)

    def test_assess_portfolio_community_8(self):
        params = {
            'start_date': dt.datetime(2002,2,9),
            'end_date': dt.datetime(2010,10,22),
            'symbols': ['CCT','JNJ','ERTS','MCO','R','WDC','BBT','JOY','PLL'],
            'allocations': [0.2, 0.2, 0.1, 0.1, 0.2, 0, 0, 0.2, 0],
            'start_value': 1000000,
            'risk_free_rate': 0.09,
            'sample_freq': 252
        }

        results = {
            'cumulative_return': 2.23953074157,
            'average_daily_return': 0.000752931076572,
            'volatility': 0.0207667097994,
            'sharpe_ratio': -68.2223284941,
            'end_value': 3239530.74157
        }

        self._run_test(params, results)

    def test_assess_portfolio_community_9(self):
        params = {
            'start_date': dt.datetime(2001,1,10),
            'end_date': dt.datetime(2010,11,20),
            'symbols': ['WWY','OMX','NFX','AVB','EW','JWN','CBS','SH','UNH','CCI'],
            'allocations': [0.2, 0.1, 0.1, 0.1, 0.2, 0, 0.1, 0, 0.2, 0],
            'start_value': 1000000,
            'risk_free_rate': 0.1,
            'sample_freq': 252
        }

        results = {
            'cumulative_return': 2.07523973914,
            'average_daily_return': 0.000526285338006,
            'volatility': 0.0121137423831,
            'sharpe_ratio': -130.355774126,
            'end_value': 3075239.73914
        }

        self._run_test(params, results)

    def _run_test(self, params, results):
        start_date = params['start_date']
        end_date = params['end_date']
        symbols = params['symbols']
        allocations = params['allocations']
        start_value = params['start_value']
        risk_free_rate = params['risk_free_rate']
        sample_freq = params['sample_freq']

        cumulative_return, \
        average_daily_return, \
        volatility, \
        sharpe_ratio, \
        end_value = analysis.assess_portfolio(sd=start_date, ed=end_date, syms=symbols, allocs=allocations,
                                              sv=start_value, rfr=risk_free_rate, sf=sample_freq, gen_plot=False)

        self.assertAlmostEqual(cumulative_return, results['cumulative_return'], delta=1e-8,
                               msg="Incorrect cumulative return {}, expected to be {}".format(
                                   cumulative_return, results['cumulative_return']))

        self.assertAlmostEqual(average_daily_return, results['average_daily_return'], delta=1e-8,
                               msg="Incorrect avg. daily return {}, expected to be {}".format(
                                   average_daily_return, results['average_daily_return']))

        self.assertAlmostEqual(volatility, results['volatility'], delta=1e-8,
                               msg="Incorrect volatility {}, expected to be {}".format(
                                   volatility, results['volatility']))

        self.assertAlmostEqual(sharpe_ratio, results['sharpe_ratio'], delta=1e-8,
                               msg="Incorrect sharpe ratio {}, expected to be {}".format(
                                   sharpe_ratio, results['sharpe_ratio']))

        if results['end_value']:
            # The reason behind lowering the delta to 1e-5 here is
            # because portfolio values are given only up to the 5th decimal
            self.assertAlmostEqual(end_value, results['end_value'], delta=1e-5,
                                   msg="Incorrect end value {}, expected to be {}".format(
                                       end_value, results['end_value']))

if __name__ == '__main__':
    unittest.main(verbosity=2)
