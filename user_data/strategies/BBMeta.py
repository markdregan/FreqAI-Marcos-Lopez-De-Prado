##
# Brief: Proof of concept project to build a toy primary model & train a secondary meta model
# on top of it. Key part of this exercise is to generate labeled data that can be used by the
# secondary meta-model - specifically: 1 = upper barrier profit threshold hit, 0 = vertical
# time elapsed barrier hit OR bottom stop loss barrier hit. In freqtrade, custom_sell
# method is needed for the vertical barrier but minimal_roi is usable for the upper barrier.
# Additionally, the meta-model needs additional TA / regime signals which need to be produced
# by populate_indicators.
# Author: markdregan@gmail.com

from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, merge_informative_pair
from freqtrade.vendor.qtpylib import indicators as qtpylib
from joblib import load
from pandas import DataFrame
from pathlib import Path
from ta import add_all_ta_features

import numpy as np  # noqa
import pandas as pd  # noqa


class BBPrimary(IStrategy):
    """Simple toy Bollinger Band Strategy with Meta Model for testing purposes."""

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 2

    # Required. But effectively disabled.
    minimal_roi = {'1000': 10.0}

    # Stop loss disabled. Use custom_sell() for TBM meta labeling.
    stoploss = -1.0

    # use_custom_stoploss = True

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Hyperoptable parameters
    # Note: would be interesting to use hyperopt to explore various options for
    # upper, lower and vertical barriers.
    # buy_rsi = IntParameter(low=1, high=50, default=30, space='buy', optimize=True, load=True)
    # sell_rsi = IntParameter(low=50, high=100, default=70, space='sell', optimize=True, load=True)

    # Timeframe for the strategy.
    timeframe = '5m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 100

    # Optional order type mapping.
    order_types = {
        'buy': 'market',
        'sell': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    # GTC (Good till canceled)
    # FOK (Fill Or Kill)
    # IOC (Immediate Or Canceled)
    order_time_in_force = {'buy': 'gtc', 'sell': 'gtc'}

    plot_config = {
        'main_plot': {
            'volatility_bbh': {
                'color': 'grey'
            },
            'volatility_bbl': {
                'color': 'grey'
            }
        },
        'subplots': {
            'MetaModel': {
                'meta_model_proba': {
                    'color': 'yellow'
                },
            },
        }
    }

    def informative_pairs(self):

        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()

        # Assign tf to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, '1h') for pair in pairs]
        informative_pairs += [(pair, '1d') for pair in pairs]
        # Add additional informative pairs
        informative_pairs += [
            ("BTC/USDT", "1h"),
            ("BTC/USDT", "1d"),
        ]

        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame,
                            metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame. For experimenting purposes,
        all TA signals are added to the data frame. These are used buy the meta model as
        regime/market signals when predicting if the primary model is making a good or bad decision.
            ----------
            dataframe: Dataframe with data from the exchange
            metadata: Additional information, like the currently traded pair
            return: a Dataframe with all mandatory indicators for the strategies
        """
        """
        if not self.dp:
            # Don't do anything if DataProvider is not available.
            return dataframe"""

        # Apply TA indicators to the pair dataframe
        dataframe = add_all_ta_features(dataframe,
                                        open="open",
                                        high="high",
                                        low="low",
                                        close="close",
                                        volume="volume",
                                        fillna=True)

        # Get lower res OHLCV data for pair in question
        low_res_tf = '1h'
        dataframe_low_res = self.dp.get_pair_dataframe(pair=metadata['pair'],
                                                       timeframe=low_res_tf)
        dataframe_low_res = add_all_ta_features(dataframe_low_res,
                                                open="open",
                                                high="high",
                                                low="low",
                                                close="close",
                                                volume="volume",
                                                fillna=False)

        # Add BTC 1h informative pair
        btc_high_tf = '1h'
        dataframe_btc_1h = self.dp.get_pair_dataframe(pair='BTC/USDT',
                                                      timeframe=btc_high_tf)
        dataframe_btc_1h = add_all_ta_features(dataframe_btc_1h,
                                               open="open",
                                               high="high",
                                               low="low",
                                               close="close",
                                               volume="volume",
                                               fillna=False)
        skip_columns = ['date', 'pair']
        dataframe_btc_1h.columns = [
            x if x in skip_columns else x + '_btc'
            for x in dataframe_btc_1h.columns
        ]

        # Add BTC 1d informative pair
        btc_low_tf = '1d'
        dataframe_btc_1d = self.dp.get_pair_dataframe(pair='BTC/USDT',
                                                      timeframe=btc_low_tf)
        dataframe_btc_1d = add_all_ta_features(dataframe_btc_1d,
                                               open="open",
                                               high="high",
                                               low="low",
                                               close="close",
                                               volume="volume",
                                               fillna=False)
        dataframe_btc_1d.columns = [
            x if x in skip_columns else x + '_btc'
            for x in dataframe_btc_1d.columns
        ]

        # Merge Informative Pairs
        dataframe = merge_informative_pair(dataframe,
                                           dataframe_low_res,
                                           self.timeframe,
                                           low_res_tf,
                                           ffill=True)
        dataframe = merge_informative_pair(dataframe,
                                           dataframe_btc_1h,
                                           self.timeframe,
                                           btc_high_tf,
                                           ffill=True)
        dataframe = merge_informative_pair(dataframe,
                                           dataframe_btc_1d,
                                           self.timeframe,
                                           btc_low_tf,
                                           ffill=True)

        # Add reference to pair so ML model can generate feature (OHE) for prediction
        dataframe['pair_copy'] = metadata['pair']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame,
                           metadata: dict) -> DataFrame:
        """
        Toy buy strategy: buy when bollinger band lower is crossed below.
            ----------
            dataframe: DataFrame populated with indicators
            metadata: Additional information, like the currently traded pair
            return: DataFrame with buy column
        """

        dataframe.loc[(
            # Close crosses below bb lower
            (qtpylib.
             crossed_below(dataframe['open'], dataframe['volatility_bbl'])) &
            # Ensure this candle had volume (important for backtesting)
            (dataframe['volume'] > 0)),
                      ['buy', 'buy_tag']] = (1, 'bbl_crossed')

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame,
                            metadata: dict) -> DataFrame:
        """
        Toy sell strategy: buy when bollinger band higher is crossed above.
            ----------
            dataframe: DataFrame populated with indicators
            metadata: Additional information, like the currently traded pair
            return: DataFrame with sell column
        """
        """
        dataframe.loc[
            (
                # Close crosses above bb upper
                (qtpylib.crossed_above(dataframe['close'], dataframe['volatility_bbh'])) &
                # Ensure this candle had volume (important for backtesting)
                (dataframe['volume'] > 0)
            ),
            ['sell', 'sell_tag']] = (1, 'bbh_crossed')
        """
        return dataframe

    def custom_sell(self, pair: str, trade: Trade, current_time, current_rate: float,
                    current_profit: float, **kwargs):
        """Custom sell needed to sell all trades open after X minutes & to label that forced
        sale as such. minimal_roi was not suitable for this because it would not allow
        labeling of the forced sell reason (ie. vertical barrier vs upper barrier hit.
            ----------
            pair : Pair that's currently analyzed
            trade : trade object
            current_time : datetime object, containing the current datetime
            current_rate : Current price, calculated based on pricing settings in ask_strategy
            current_profit : Current profit (as ratio), calculated based on current_rate
            return : custom sell_reason
        """

        # Sell any positions at a loss if they are held for more than two hours.
        if (current_time - trade.open_date_utc).seconds / 3600 >= 6:
            return 'vertical_barrier_force_sell'

        # Triple Barrier Method: Upper and lower barriers based on EMA Daily Volatility
        # Obtain pair dataframe
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        # Upper / lower barrier multiplier
        PT_MULTIPLIER = 1.03
        SL_MULTIPLIER = 0.985

        # Look up original candle on the trade date
        trade_date = timeframe_to_prev_date(self.timeframe,
                                            trade.open_date_utc)
        trade_candle = dataframe.loc[dataframe['date'] == trade_date]

        # trade_candle may be empty for trades that just opened as it is still incomplete.
        if not trade_candle.empty:
            trade_candle = trade_candle.squeeze()

            # Profit taking upper barrier sell trigger
            if current_rate > trade_candle['open'] * PT_MULTIPLIER:
                return 'upper_barrier_pt_sell'

            # Stop loss lower barrier sell trigger
            elif current_rate < trade_candle['open'] * SL_MULTIPLIER:
                return 'lower_barrier_sl_sell'

        return None


class BBMeta(BBPrimary):
    """Subclass of BBPrimary that combines both primary strategy and secondary meta model.
    The main extensions are to a) load the ML model, b) make predictions and add to data
    frame & c) make buy_trend decisions based on primary strategy signals & meta model."""

    # Define profit taking for Meta Model
    minimal_roi = {'0': 0.3}

    # Stop loss disabled. Use custom_sell() for TBM meta labeling.
    stoploss = -0.015

    # Disable custom sell function used for generating labeled data
    use_sell_signal = False

    def __init__(self, *args, **kwargs):
        """Load the ML MetaModel so we can make predictions. Need to ensure
        __init()__ from super class is executed."""

        # Need to load config via __init__() from super class
        super().__init__(*args, **kwargs)

        # Load ML model & config details when strategy is instantiated
        filename = 'BBMeta'
        filepath = Path('user_data', 'meta_model', filename)
        model_and_config = load(open(filepath, 'rb'))
        self.clf = model_and_config['model']
        self.model_threshold = model_and_config['model_threshold']
        self.model_X_features = model_and_config['X_features']

    def populate_indicators(self, dataframe: DataFrame,
                            metadata: dict) -> DataFrame:
        """Extend the populate_indicators method from super class to add prediction
        probabilities from the meta model."""

        # Extend the super class to add additional signals
        dataframe = super().populate_indicators(dataframe, metadata)

        # Add predictions from the meta model to dataframe
        dataframe['meta_model_proba'] = self.clf.predict_proba(
            dataframe.loc[:, self.model_X_features])[:, 1]

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame,
                           metadata: dict) -> DataFrame:
        """Combine primary rule and meta model prediction to make buy determination.
        This method over-rides the supper class method.
            ----------
            dataframe: DataFrame populated with indicators
            metadata: Additional information, like the currently traded pair
            return: DataFrame with buy column"""

        dataframe.loc[(
            # Close crosses below bb lower
            (qtpylib.
             crossed_below(dataframe['open'], dataframe['volatility_bbl'])) &
            # If meta model probability is above threshold
            (dataframe['meta_model_proba'] > self.model_threshold) &
            # Ensure this candle had volume (important for backtesting)
            (dataframe['volume'] > 0)),
                      ['buy', 'buy_tag']] = (1, 'meta_model_buy')

        return dataframe
