"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
"""

__author__ = "Allan Reyes"
__email__ = "reyallan@gatech.edu"
__userId__ = "arx3"

import datetime as dt
import numpy as np
import pandas as pd
import util as ut
import random

from indicators import bollinger_bands, momentum, money_flow_index, simple_moving_average
from marketsimcode import compute_portvals
from QLearner import QLearner

class StockData(object):
    """Holds stock information"""
    def __init__(self, symbol, start_date, end_date):
        """
        Args:
            - symbol: The symbol to get stock information for
            - start_date: datetime
                          The start date of the stock information to hold
            - end_date: datetime
                        The end date of the stock information to hold
        """
        self._symbol = symbol
        self._dates = pd.date_range(start_date, end_date)
        self._price = None
        self._high = None
        self._low = None
        self._volume = None

        self._fetch_data()

    @property
    def price(self):
        """The adjusted close price of the stock"""
        return self._price

    @property
    def high(self):
        """The high price of the stock"""
        return self._high

    @property
    def low(self):
        """The low price of the stock"""
        return self._low

    @property
    def volume(self):
        """The volume of the stock"""
        return self._volume

    @property
    def trading_dates(self):
        """The dates that the market traded"""
        return self._price.index

    def _fetch_data(self):
        self._price = self._get_data('Adj Close')
        self._high = self._get_data('High')
        self._low = self._get_data('Low')
        self._volume = self._get_data('Volume')

    def _get_data(self, attribute):
        # Automatically adds the SPY
        data = ut.get_data([self._symbol], self._dates, colname=attribute)
        data.fillna(method='ffill', inplace=True)
        data.fillna(method='bfill', inplace=True)

        # Only get the data for the stock in our portfolio
        data = data[[self._symbol]]

        return data

# TODO: This class should be refactored into
#       a factory of concrete Discretizers
class IndicatorDiscretizer(object):
    @property
    def momentum_max_bucket(self):
        """The largest bucket for the Momentum indicator"""
        return 4

    @property
    def simple_moving_average_max_bucket(self):
        """The largest bucket for the Simple Moving Average indicator"""
        return 4

    @property
    def bollinger_bands_max_bucket(self):
        """The largest bucket for the Bollinger Bands indicator"""
        return 4

    @property
    def money_flow_index_max_bucket(self):
        """The largest bucket for the Money Flow Index indicator"""
        return 3

    """Discretizes indicators"""
    def momentum(self, mtm):
        """Discretizes the Momentum indicator"""
        # Typical range for momentum: -0.5 to 0.5
        # Five buckets:
        # 1. x < -0.5
        # 2. -0.5 <= x <= 0.0
        # 3. 0.0 < x <= 0.5
        # 4. x > 0.5
        # 5. x == NaN
        discretized = mtm.copy()
        discretized.values[mtm < -0.5] = 0
        discretized.values[(mtm >= -0.5) & (mtm <= 0.0)] = 1
        discretized.values[(mtm > 0.0) & (mtm <= 0.5)] = 2
        discretized.values[mtm > 0.5] = 3
        discretized.values[mtm.isnull()] = 4

        return discretized.astype('int32')

    def simple_moving_average(self, sma):
        """Discretizes the Simple Moving Average indicator"""
        # Typical range for sma: -0.5 to 0.5
        # Five buckets:
        # 1. x < -0.5
        # 2. -0.5 <= x <= 0.0
        # 3. 0.0 < x <= 0.5
        # 4. x > 0.5
        # 5. x == NaN
        discretized = sma.copy()
        discretized.values[sma < -0.5] = 0
        discretized.values[(sma >= -0.5) & (sma <= 0.0)] = 1
        discretized.values[(sma > 0.0) & (sma <= 0.5)] = 2
        discretized.values[sma > 0.5] = 3
        discretized.values[sma.isnull()] = 4

        return discretized.astype('int32')

    def bollinger_bands(self, bbands):
        """Discretizes the Bollinger Bands indicators"""
        # Typical range for bbands: -1.0 to 1.0
        # Five buckets:
        # 1. x < -1.0
        # 2. -1.- <= x <= 0.0
        # 3. 0.0 < x <= 1.0
        # 4. x > 1.0
        # 5. x == NaN
        discretized = bbands.copy()
        discretized.values[bbands < -1.0] = 0
        discretized.values[(bbands >= -1.0) & (bbands <= 0.0)] = 1
        discretized.values[(bbands > 0.0) & (bbands <= 1.0)] = 2
        discretized.values[bbands > 1.0] = 3
        discretized.values[bbands.isnull()] = 4

        return discretized.astype('int32')

    def money_flow_index(self, mfi):
        """Discretizes the Money Flow Index indicator"""
        # Typical range for mfi: 0 to 100
        # TODO: Consider using 10 buckets instead
        # Four buckets
        # 1. x < 30 (oversold bucket)
        # 2. 30 <= x <= 70 (no indication)
        # 3. x > 70 (overbought bucket)
        # 4. x == NaN
        discretized = mfi.copy()
        discretized.values[mfi < 30] = 0
        discretized.values[(mfi >= 30) & (mfi <= 70)] = 1
        discretized.values[mfi > 70] = 2
        discretized.values[mfi.isnull()] = 3

        return discretized.astype('int32')

class TradingStateFactory(object):
    """Factory that creates trading states
       from underlying technical indicators
    """
    def __init__(self, stock_data, indicator_discretizer, lookback=10):
        """
        Args:
            - stock_data: StockData
                          A StockData instance
            - indicator_discretizer: IndicatorDiscretizer
                                     An IndicatorDiscretizer instance
            - lookback: int
                        The days to look back for the underlying indicators
                        Defaults to 10 days
        """
        self._stock_data = stock_data
        self._indicator_discretizer = indicator_discretizer
        self._lookback = lookback
        self._num_states = None
        self._indicators = None

        self._compute_number_of_states()
        self._compute_indicators()

    @property
    def num_states(self):
        """The total number of states of all underlying trading states"""
        return self._num_states

    def create(self, day):
        """
        Creates a trading state for a particular day

        Args:
            - day: datetime
                   The day to create a trading state for

        Returns:
            - An int representing the trading state
        """
        return self._indicators.loc[day]

    def _compute_number_of_states(self):
        # The total number of states is the largest number we can build
        # from all the max buckets of all indicators
        all_buckets = [
            self._indicator_discretizer.momentum_max_bucket, \
            self._indicator_discretizer.simple_moving_average_max_bucket, \
            self._indicator_discretizer.bollinger_bands_max_bucket, \
            self._indicator_discretizer.money_flow_index_max_bucket
        ]
        largest_number = int(''.join(map(str, all_buckets)))

        # Add 1 to account for zero-indexing and allow the largets number to
        # be used as an index too
        self._num_states = largest_number + 1

    def _compute_indicators(self):
        price = self._stock_data.price
        high = self._stock_data.high
        low = self._stock_data.low
        volume = self._stock_data.volume

        mtm = momentum(price, self._lookback)
        sma, sma_ratio = simple_moving_average(price, self._lookback)
        bbands = bollinger_bands(price, sma, self._lookback)
        mfi = money_flow_index(price, high, low, volume, self._lookback)

        self._indicators = self._discretize((mtm, sma_ratio, bbands, mfi))

    def _discretize(self, indicators):
        mtm, sma, bbands, mfi = indicators

        discretized_mtm = self._indicator_discretizer.momentum(mtm)
        discretized_sma = self._indicator_discretizer.simple_moving_average(sma)
        discretized_bbands = self._indicator_discretizer.bollinger_bands(bbands)
        discretized_mfi = self._indicator_discretizer.money_flow_index(mfi)

        # Concat all indicators into a single dataframe horizontally
        discretized_indicators = pd.concat(
            [discretized_mtm, discretized_sma, discretized_bbands, discretized_mfi],
            axis=1
        )

        # Convert them into a string to then parse them as an int
        # See: https://stackoverflow.com/questions/489999/convert-list-of-ints-to-one-number
        discretized_indicators = discretized_indicators.apply(
            lambda row: int(''.join(map(str, row))), axis=1
        )

        return discretized_indicators

class TradingEnvironment(object):
    """Encapsulates trading as a Reinforcement Learning environment"""
    def __init__(self):
        self._qlearner = None
        self._trading_state_factory = None
        self._stock_data = None
        self._trading_options = None

        # Maintains a mapping of actions to labels
        self._action_mapping = {
            0: 'LONG',
            1: 'CASH',
            2: 'SHORT'
        }

    @property
    def qlearner(self):
        return self._qlearner

    @qlearner.setter
    def qlearner(self, qlearner):
        """Sets the Q-learning agent to use"""
        self._qlearner = qlearner

    @property
    def trading_state_factory(self):
        return self._trading_state_factory

    @trading_state_factory.setter
    def trading_state_factory(self, trading_state_factory):
        """Sets an instance of a TradingStateFactory"""
        self._trading_state_factory = trading_state_factory

    @property
    def stock_data(self):
        return self._stock_data

    @stock_data.setter
    def stock_data(self, stock_data):
        """Sets an instance of a StockData"""
        self._stock_data = stock_data

    @property
    def trading_options(self):
        return self._trading_options

    @trading_options.setter
    def trading_options(self, trading_options):
        """Sets a dictionary of trading options.

        Available options:
            - trading_dates: list(datetime)
            - impact: float
        """
        self._trading_options = trading_options

    def run_learning_episode(self):
        """Runs a single instance of a learning episode

        Returns:
            - A DataFrame of orders that should be executed
        """
        # Our current holding
        holding = None
        trading_dates = self._trading_options['trading_dates']
        # Holds the orders that the agent decides should be taken
        orders = pd.DataFrame(index=trading_dates, columns=['Shares'])

        # Initialize the learner with the first state for the first day
        state = self._trading_state_factory.create(trading_dates[0])
        self._qlearner.querysetstate(state)

        for index, date in enumerate(trading_dates):
            # Use the current date (i.e. today) as yesterday for the first trading day
            yesterday = trading_dates[index - 1] if index > 0 else date

            # This is a little different from "common" implementations
            # of Q-Learning agents because, in the case of trading, we need
            # to compute the reward first each day since it would allow
            # us to see how our last action (i.e. transaction) did from
            # yesterday to today
            reward = self._reward(date, yesterday, holding)

            # Get an action for the state and learn!
            action = self._qlearner.query(state, reward)
            # Execute the action to update our holding position
            order, holding = self._execute_action(action, holding)
            orders.loc[date] = order

            # Are we done?
            if index == len(trading_dates) - 1:
                return orders

            # Get the next state (i.e. for tomorrow) if we are not done
            state = self._trading_state_factory.create(trading_dates[index + 1])

    def run_interaction_episode(self):
        """Runs a single instance of an interaction episode where no learning occurs

        Returns:
            - A DataFrame of orders that should be executed
        """
        # Our current holding
        holding = None
        trading_dates = self._trading_options['trading_dates']
        # Holds the orders that the agent decides should be taken
        orders = pd.DataFrame(index=trading_dates, columns=['Shares'])

        for index, date in enumerate(trading_dates):
            state = self._trading_state_factory.create(date)
            action = self._qlearner.querysetstate(state)
            order, holding = self._execute_action(action, holding)
            orders.loc[date] = order

        return orders

    def _reward(self, today, yesterday, holding):
        if holding == 'CASH' or holding is None:
            # We don't have any shares, so there is no reward
            return 0.

        price_today = self._apply_impact(self._stock_data.price.loc[today], holding)
        price_yesterday = self._stock_data.price.loc[yesterday]

        # Reward is the daily return times the position we are currently holding
        # where LONG = 1, and SHORT = -1 to account for price drops being desired
        # when shorting
        daily_return = (price_today / price_yesterday) - 1.
        multiplier = 1. if holding == 'LONG' else -1.

        return daily_return * multiplier

    def _apply_impact(self, price, holding):
        impact = self._trading_options['impact']

        # Market impact affects the price of the stock in different
        # ways depending on what our current position is. This comes
        # from the fact that market impact is used to account for
        # unexpected changes in the market which, for the purposes of
        # developing trading strategies, it should always go against us
        if holding == 'LONG':
            # We are currently holding some shares, so the price goes
            # against us by dropping unexpectedly which will affect
            # our return
            return price * (1. - impact)

        if holding == 'SHORT':
            # We are currently shorting some shares, so the price goes
            # against us by increasing unexpectedly which will affect
            # our return
            return price * (1. + impact)

        return price

    def _execute_action(self, action, holding):
        # Executing an action generates an order and updates our holdings
        action_label = self._action_mapping[action]

        if action_label == 'LONG':
            return self._execute_long(holding)
        elif action_label == 'SHORT':
            return self._execute_short(holding)
        elif action_label == 'CASH':
            return self._execute_cash(holding)
        else:
            raise ValueError("Unrecognized action: {}".format(action_label))

    def _execute_long(self, holding):
        # No-op if we are already in a long position
        if holding == 'LONG':
            return 0, 'LONG'

        to_buy = 1000
        # If we are currently shorting, we should buy double
        # to reach the long position
        if holding == 'SHORT':
            to_buy = 2000

        return to_buy, 'LONG'

    def _execute_short(self, holding):
        # No-op if we are already in a short position
        if holding == 'SHORT':
            return 0, 'SHORT'

        to_sell = 1000
        # If we were longing, sell double to reach the short position
        if holding == 'LONG':
            to_sell = 2000

        return to_sell * -1, 'SHORT'

    def _execute_cash(self, holding):
        # If we have a long position, then we should sell to close
        if holding == 'LONG':
            return -1000, 'CASH'

        # If we have a short position, then we should buy to close
        if holding == 'SHORT':
            return 1000, 'CASH'

        # No-op if we are already in a "cash" position
        return 0, 'CASH'

class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False, impact=0.0):
        self.verbose = verbose
        self.impact = impact

        self._learner = None
        self._indicator_discretizer = IndicatorDiscretizer()
        self._trading_environment = TradingEnvironment()

        self._metadata = {}

    @property
    def metadata(self):
        """Returns metadata about this StrategyLearner"""
        return self._metadata

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000):

        stock_data = StockData(symbol, sd, ed)
        trading_state_factory = TradingStateFactory(stock_data, self._indicator_discretizer)

        self._learner = QLearner(
            num_states=trading_state_factory.num_states,
            num_actions=3,
            alpha=0.2,
            gamma=0.9,
            rar=0.5,
            radr=0.99,
            dyna=0
        )

        self._trading_environment.qlearner = self._learner
        self._trading_environment.trading_state_factory = trading_state_factory
        self._trading_environment.stock_data = stock_data
        self._trading_environment.trading_options = {
            'trading_dates': stock_data.trading_dates,
            'impact': self.impact
        }

        latest_cumulative_return = -999
        current_cumulative_return = 0
        episodes = 0

        # Run learning episodes until the cumulative return of the strategy has converged
        while np.abs(latest_cumulative_return - current_cumulative_return) > 0.001:
            latest_cumulative_return = current_cumulative_return

            trades = self._trading_environment.run_learning_episode()
            orders = self._convert_trades_to_marketisim_orders(symbol, trades)

            portfolio_values = compute_portvals(
                orders,
                start_val=sv,
                commission=0.,
                impact=self.impact,
                prices=stock_data.price.copy(),
            )

            current_cumulative_return = self._compute_cumulative_return(portfolio_values)

            episodes += 1

        # Keep track of the number of training episodes
        self._metadata['training_episodes'] = episodes

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):

        stock_data = StockData(symbol, sd, ed)
        trading_state_factory = TradingStateFactory(stock_data, self._indicator_discretizer)

        self._trading_environment.trading_state_factory = trading_state_factory
        self._trading_environment.stock_data = stock_data
        self._trading_environment.trading_options = {
            'trading_dates': stock_data.trading_dates,
            'impact': self.impact
        }

        trades = self._trading_environment.run_interaction_episode()

        # Keep track of the total number of entries generated
        self._metadata['entries'] = self._count_total_number_of_entries(trades)

        return trades

    def _convert_trades_to_marketisim_orders(self, symbol, trades):
        # Convert the trades into the format expected by my marketsimcode.py
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

    def _compute_cumulative_return(self, portfolio_values):
        return (portfolio_values[-1] / portfolio_values[0]) - 1

    def _count_total_number_of_entries(self, trades):
        # Entries are any trades were the strategy suggests
        # either going long (positive) or shorting (negative)
        return trades.values[trades != 0].shape[0]

if __name__=="__main__":
    print "One does not simply think up a strategy"