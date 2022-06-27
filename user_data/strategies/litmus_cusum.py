# Define kama strategy that is imported into a central Litmus Strategy

from user_data.litmus import indicator_helpers

import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas as pd


def plot_config():
    """Define plot config for strategy"""

    plot_config = {}
    plot_config['main_plot'] = {

    }
    plot_config['subplots'] = {
        "+CUSUM": {
            "cusum_pos_threshold": {
                "color": "grey"
            },
            "cusum_s_pos": {
                "color": "green"
            }
        },
        "-CUSUM": {
            "cusum_neg_threshold": {
                "color": "grey"
            },
            "cusum_s_neg": {
                "color": "red"
            }
        },
        "Trigger": {
            "cusum_trigger": {
                "color": "pink"
            }
        }
    }

    return plot_config


def populate_indicators(dataframe) -> pd.DataFrame:
    """Features/columns needed to support strategy"""

    dataframe = indicator_helpers.daily_volatility(dataframe, shift=288, lookback=30)
    dataframe = indicator_helpers.cusum_filter(dataframe, threshold_coeff=0.6)

    return dataframe


def populate_entry_trend(dataframe) -> pd.DataFrame:
    """Logic to define entry positions in strategy"""

    dataframe.loc[
        (qtpylib.crossed_above(dataframe['entry_trigger'], 0.5))
        & (dataframe['volume'] > 0),
        ['enter_long', 'enter_tag']] = (1, 'enter_long_trigger')

    return dataframe
