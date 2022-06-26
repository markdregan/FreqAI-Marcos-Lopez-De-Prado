##
# Brief:    This class gets signals from a freqtrade strategy and outcomes of the trades
#           from that strategy backtest. It then joins together and returns the DataFrame.
# Author: markdregan@gmail.com

from pathlib import Path

import joblib
import json
import pandas as pd


class GetLabeledData:

    def __init__(self, timeframe: str, bt_filename: str = '',
                 bt_pkl_filename: str = ''):

        self.bt_dir = Path('user_data', 'backtest_results')
        self.timeframe = timeframe

        self.bt_filename = bt_filename
        self.bt_pkl_filename = bt_pkl_filename

    def get_signals(self) -> dict:
        """Load and unpickle file from backtest. Concat into one DataFrame"""

        file_path = Path(self.bt_dir, self.bt_pkl_filename)
        bt = joblib.load(open(file_path, mode='rb'))

        temp_df = []

        for strategy, signal_dict in bt.items():
            for pair, df in signal_dict.items():
                df['pair'] = pair
                df['strategy'] = strategy
                df['strategy_copy'] = strategy
                temp_df.append(df)

        df = pd.concat(temp_df)
        df = df.set_index(['strategy', 'pair', 'date'])
        df.index.rename(names=['strategy', 'pair', 'date'], inplace=True)
        df.sort_index(inplace=True)

        return df

    def _get_prev_candle_timestamp(self) -> int:
        """Get the timestamp of the previous candle based on timeframe of strategy"""

        timeframe_to_minute_offset_map = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '6h': 360,
            '12h': 720,
            '1d': 1440
        }

        return timeframe_to_minute_offset_map[self.timeframe]

    def get_trade_outcomes(self) -> pd.DataFrame:
        """Load backtest results from file and get the outcomes of all trades.
        Note Backtest() needs to have been run before using GetLabeledData()."""

        file = open(Path(self.bt_dir, self.bt_filename))
        trade_outcomes_json = json.load(file)

        temp = []

        for strategy, trade_dict in trade_outcomes_json['strategy'].items():
            strategy_df = pd.DataFrame(trade_dict['trades'])
            strategy_df['strategy'] = strategy
            temp.append(strategy_df)

        df = pd.concat(temp)

        df['open_date'] = pd.to_datetime(df['open_date'], errors='coerce', utc=True)
        prev_candle_offset = self._get_prev_candle_timestamp()
        df['date'] = df['open_date'] - pd.Timedelta(prev_candle_offset, 'minute')

        df.set_index(keys=['strategy', 'pair', 'date'],
                     drop=True, inplace=True, verify_integrity=True)
        df.sort_index(inplace=True)

        return df

    def get_all_data(self) -> pd.DataFrame:
        """Get the trade outcomes for all buy triggers for all pairs in a
        given strategy."""

        # Gather the buy indicators + outcome data before joining
        signals = self.get_signals()
        trade_outcomes = self.get_trade_outcomes()

        # Join together
        results = pd.merge(left=signals, right=trade_outcomes,
                           how='outer', left_index=True, right_index=True)

        # Sort index so pd.IndexSlice works
        results.sort_index(inplace=True)

        return results
