# Copy of Vulcan Strategy with MetaModel added
# Author: markdregan@gmail.com

from freqtrade.persistence import Trade
from joblib import load
from pandas import DataFrame
from pathlib import Path
from user_data.strategies.VulcanPrimary import VulcanPrimary

import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np  # noqa
import pandas as pd  # noqa

master_plot_config = {
    "main_plot": {
        "SMA": {
            "color": "red"
        },
    },
    "subplots": {
        "Meta": {
            "meta_model_proba": {
                "color": "yellow"
            }
        },
        "Buy & Sell": {
            "buy_trigger": {
                "color": "green"
            },
            "sell_trigger": {
                "color": "red"
            },
        },
        "Grow & Shrink": {
            "growing_SMA": {
                "color": "green"
            },
            "shrinking_SMA": {
                "color": "red"
            },
        },
        "RSI": {
            "RSI": {
                "color": "green"
            },
            "RSI_SMA": {
                "color": "purple"
            },
        },
        "STOCH": {
            "slowd": {
                "color": "green"
            },
            "slowk": {
                "color": "purple"
            },
        },
    },
}


class VulcanMeta(VulcanPrimary):

    INTERFACE_VERSION = 3

    stoploss = -1
    timeframe = "12h"
    minimal_roi = {"0": 3.0}

    # Turn on position adjustment
    position_adjustment_enable = False

    plot_config = master_plot_config

    def __init__(self, *args, **kwargs):
        """Load the ML MetaModel so we can make predictions. Need to ensure
        __init()__ from super class is executed."""

        # Need to load config via __init__() from super class
        super().__init__(*args, **kwargs)

        # Load ML model & config details when strategy is instantiated
        filename = 'VulcanMeta.pkl'
        filepath = Path('user_data', 'meta_model', filename)
        model_and_config = load(open(filepath, 'rb'))
        self.clf = model_and_config['model']
        self.model_threshold = model_and_config['model_threshold']
        self.model_X_features = model_and_config['X_features']

    def populate_indicators(self, dataframe: DataFrame,
                            metadata: dict) -> DataFrame:

        """Extend the populate_indicators method from super class to add prediction
        probabilities from the metamodel."""

        # Extend the super class to add additional signals
        dataframe = super().populate_indicators(dataframe, metadata)

        # Add predictions from the meta model to dataframe
        dataframe['meta_model_proba'] = self.clf.predict_proba(
            dataframe.loc[:, self.model_X_features])[:, 1]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame,
                             metadata: dict) -> DataFrame:
        """
        Buy trend indicator when buy_trigger crosses from 0 to 1
            ----------
            dataframe: DataFrame populated with indicators
            metadata: Additional information, like the currently traded pair
            return: DataFrame with buy column
        """

        dataframe.loc[
            (qtpylib.crossed_above(dataframe['buy_trigger'], 0.5))
            # If meta model probability is above threshold
            & (dataframe['meta_model_proba'] > 0.2)
            # Volume > 0 for backtesting
            & (dataframe['volume'] > 0),
            ['enter_long', 'enter_tag']] = (1, 'primary_buy_trigger')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame,
                            metadata: dict) -> DataFrame:
        """
        Disable populate sell
            ----------
            dataframe: DataFrame populated with indicators
            metadata: Additional information, like the currently traded pair
            return: DataFrame with sell column
        """

        dataframe.loc[
            (qtpylib.crossed_above(dataframe['sell_trigger'], 0.5))
            & (dataframe['volume'] > 0),
            ['exit_long', 'exit_tag']] = (1, 'main_sell_trigger')

        return dataframe

    def adjust_trade_position(self, trade: Trade, current_time,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):

        filled_buys = trade.select_filled_orders('buy')

        try:
            # This returns first order stake size
            stake_amount = filled_buys[0].cost
            return stake_amount
        except Exception as exception:
            print(exception)
            pass
