# KAMA Primary Strategy
# Tradingview: https://www.tradingview.com/chart/WeEVLg4V/
# Author: markdregan@gmail.com

from collections import defaultdict

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame
from user_data.litmus import indicator_helpers
from user_data.litmus.glassnode import download_data

import litmus_kama as litmus
import ta.momentum

# Prevent pandas complaining with future warning errors
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class KamaPrimary(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "5m"

    # Required. But effectively disabled.
    minimal_roi = {
        # Effectively disabled. Not used for MetaModel. PT & SL levels defined in custom_sell()
        '1000': 1000
    }

    # Stop loss disabled. Use custom_exit() for TBM meta labeling.
    stoploss = -1

    # Turn on position adjustment
    position_adjustment_enable = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count = 100

    @property
    def plot_config(self):
        """Define plot config for strategy"""

        return litmus.plot_config()

    def informative_pairs(self):

        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()

        # Assign timeframe to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, '1h') for pair in pairs]
        # Add additional informative pairs
        informative_pairs += [
            ("BTC/USDT", "1h"),
            ("BTC/USDT", "5m")
        ]

        return informative_pairs

    def bot_start(self, **kwargs) -> None:
        """Instantiate things before bot starts"""

        # Glassnode class for fetching data from glassnode and sqlite db
        self.gn = download_data.GlassnodeData(api_key='22HCck9cUjuvUTrEvfc50rgcL7v',
                                              directory='user_data/data/glassnode/')

        # List of glassnode features that are shared across collection of tokens
        self.gn_f = ["gn_10m__v1_metrics_addresses_active_count",
                     "gn_10m__v1_metrics_addresses_receiving_count",
                     "gn_10m__v1_metrics_indicators_nvt",
                     "gn_10m__v1_metrics_indicators_nvts",
                     "gn_10m__v1_metrics_addresses_count",
                     "gn_10m__v1_metrics_indicators_velocity",
                     "gn_10m__v1_metrics_market_marketcap_usd",
                     "gn_10m__v1_metrics_market_price_drawdown_relative",
                     "gn_10m__v1_metrics_market_price_usd",
                     "gn_10m__v1_metrics_market_price_usd_close",
                     "gn_10m__v1_metrics_supply_current",
                     "gn_10m__v1_metrics_addresses_sending_count",
                     "gn_10m__v1_metrics_addresses_new_non_zero_count",
                     "gn_10m__v1_metrics_transactions_transfers_volume_mean",
                     "gn_10m__v1_metrics_transactions_transfers_volume_median",
                     "gn_10m__v1_metrics_transactions_transfers_volume_sum",
                     "gn_10m__v1_metrics_transactions_transfers_volume_between_exchanges_sum",
                     "gn_10m__v1_metrics_distribution_balance_exchanges",
                     "gn_10m__v1_metrics_transactions_transfers_to_exchanges_count",
                     "gn_10m__v1_metrics_transactions_transfers_from_exchanges_count",
                     "gn_10m__v1_metrics_transactions_transfers_between_exchanges_count",
                     "gn_10m__v1_metrics_transactions_transfers_volume_exchanges_net",
                     "gn_10m__v1_metrics_transactions_transfers_volume_from_exchanges_mean",
                     "gn_10m__v1_metrics_transactions_transfers_volume_from_exchanges_sum",
                     "gn_10m__v1_metrics_transactions_transfers_volume_to_exchanges_mean",
                     "gn_10m__v1_metrics_transactions_transfers_volume_to_exchanges_sum",
                     "gn_10m__v1_metrics_distribution_balance_exchanges_relative",
                     "gn_10m__v1_metrics_transactions_transfers_volume_within_exchanges_sum",
                     "gn_10m__v1_metrics_addresses_sending_to_exchanges_count",
                     "gn_10m__v1_metrics_addresses_receiving_from_exchanges_count",
                     "gn_10m__v1_metrics_transactions_transfers_count",
                     "gn_10m__v1_metrics_transactions_transfers_rate"]

        self.gn_btc_f = [
            'gn_10m__v1_metrics_blockchain_block_interval_median',
            'gn_10m__v1_metrics_market_marketcap_realized_usd',
            'gn_10m__v1_metrics_market_mvrv_z_score',
            'gn_10m__v1_metrics_market_mvrv',
            'gn_10m__v1_metrics_blockchain_block_interval_mean',
            'gn_10m__v1_metrics_transactions_count',
            'gn_10m__v1_metrics_blockchain_block_height',
            'gn_10m__v1_metrics_blockchain_block_size_mean',
            'gn_10m__v1_metrics_fees_fee_ratio_multiple',
            'gn_10m__v1_metrics_fees_volume_mean',
            'gn_10m__v1_metrics_fees_volume_median',
            'gn_10m__v1_metrics_fees_volume_sum',
            'gn_10m__v1_metrics_market_price_realized_usd',
            'gn_10m__v1_metrics_blockchain_block_count',
            'gn_10m__v1_metrics_blockchain_block_size_sum',
            'gn_10m__v1_metrics_transactions_rate',
            'gn_10m__v1_metrics_supply_revived_more_5y_sum',
            'gn_10m__v1_metrics_indicators_svl_1w_1m',
            'gn_10m__v1_metrics_addresses_min_point_1_count',
            'gn_10m__v1_metrics_addresses_min_1k_count',
            'gn_10m__v1_metrics_indicators_svl_3y_5y',
            'gn_10m__v1_metrics_indicators_svl_3m_6m',
            'gn_10m__v1_metrics_addresses_min_1_count',
            'gn_10m__v1_metrics_addresses_min_10k_count',
            'gn_10m__v1_metrics_indicators_svl_1y_2y',
            'gn_10m__v1_metrics_indicators_svl_1m_3m',
            'gn_10m__v1_metrics_indicators_svl_6m_12m',
            'gn_10m__v1_metrics_indicators_svl_1h_24h',
            'gn_10m__v1_metrics_indicators_svl_1h',
            'gn_10m__v1_metrics_indicators_svl_1d_1w',
            'gn_10m__v1_metrics_addresses_supply_balance_less_0001',
            'gn_10m__v1_metrics_addresses_min_10_count',
            'gn_10m__v1_metrics_addresses_min_100_count',
            'gn_10m__v1_metrics_addresses_supply_balance_more_100k',
            'gn_10m__v1_metrics_indicators_svl_5y_7y',
            'gn_10m__v1_metrics_indicators_svl_7y_10y',
            'gn_10m__v1_metrics_supply_revived_more_3y_sum',
            'gn_10m__v1_metrics_mining_difficulty_latest',
            'gn_10m__v1_metrics_supply_revived_more_2y_sum',
            'gn_10m__v1_metrics_supply_revived_more_1y_sum',
            'gn_10m__v1_metrics_supply_issued',
            'gn_10m__v1_metrics_addresses_supply_balance_01_1',
            'gn_10m__v1_metrics_mining_revenue_from_fees',
            'gn_10m__v1_metrics_mining_hash_rate_mean',
            'gn_10m__v1_metrics_mining_difficulty_mean',
            'gn_10m__v1_metrics_addresses_supply_balance_100_1k',
            'gn_10m__v1_metrics_indicators_svl_more_10y',
            'gn_10m__v1_metrics_indicators_cdd_supply_adjusted_binary',
            'gn_10m__v1_metrics_addresses_supply_balance_10k_100k',
            'gn_10m__v1_metrics_addresses_supply_balance_001_01',
            'gn_10m__v1_metrics_addresses_supply_balance_0001_001',
            'gn_10m__v1_metrics_addresses_supply_balance_1_10',
            'gn_10m__v1_metrics_addresses_min_point_zero_1_count',
            'gn_10m__v1_metrics_addresses_supply_balance_1k_10k',
            'gn_10m__v1_metrics_addresses_supply_balance_10_100',
            'gn_10m__v1_metrics_indicators_svl_2y_3y',
            'gn_10m__v1_metrics_indicators_cdd_supply_adjusted',
            'gn_10m__v1_metrics_derivatives_futures_funding_rate_perpetual',
            'gn_10m__v1_metrics_derivatives_futures_open_interest_crypto_margin_sum',
            'gn_10m__v1_metrics_derivatives_futures_annualized_basis_3m',
            'gn_10m__v1_metrics_derivatives_options_atm_implied_volatility_6_months',
            'gn_10m__v1_metrics_derivatives_options_atm_implied_volatility_3_months',
            'gn_10m__v1_metrics_derivatives_options_atm_implied_volatility_1_week',
            'gn_10m__v1_metrics_derivatives_options_atm_implied_volatility_1_month',
            'gn_10m__v1_metrics_derivatives_options_25delta_skew_all',
            'gn_10m__v1_metrics_derivatives_options_25delta_skew_6_months',
            'gn_10m__v1_metrics_derivatives_options_25delta_skew_3_months',
            'gn_10m__v1_metrics_derivatives_options_25delta_skew_1_week',
            'gn_10m__v1_metrics_derivatives_futures_estimated_leverage_ratio',
            'gn_10m__v1_metrics_derivatives_futures_liquidated_volume_long_mean',
            'gn_10m__v1_metrics_derivatives_options_open_interest_put_call_ratio',
            'gn_10m__v1_metrics_derivatives_futures_liquidated_volume_long_relative',
            'gn_10m__v1_metrics_derivatives_futures_liquidated_volume_long_sum',
            'gn_10m__v1_metrics_derivatives_futures_liquidated_volume_short_mean',
            'gn_10m__v1_metrics_indicators_average_dormancy_supply_adjusted',
            'gn_10m__v1_metrics_derivatives_options_25delta_skew_1_month',
            'gn_10m__v1_metrics_derivatives_futures_volume_daily_sum',
            'gn_10m__v1_metrics_derivatives_futures_volume_daily_perpetual_sum',
            'gn_10m__v1_metrics_derivatives_futures_liquidated_volume_short_sum',
            'gn_10m__v1_metrics_derivatives_futures_open_interest_cash_margin_sum',
            'gn_10m__v1_metrics_derivatives_futures_open_interest_sum',
            'gn_10m__v1_metrics_derivatives_futures_open_interest_crypto_margin_relative',
            'gn_10m__v1_metrics_derivatives_options_atm_implied_volatility_all',
            'gn_10m__v1_metrics_blockchain_utxo_spent_value_sum',
            'gn_10m__v1_metrics_derivatives_options_open_interest_sum',
            'gn_10m__v1_metrics_blockchain_utxo_spent_value_median',
            'gn_10m__v1_metrics_blockchain_utxo_count',
            'gn_10m__v1_metrics_blockchain_utxo_created_count',
            'gn_10m__v1_metrics_blockchain_utxo_created_value_mean',
            'gn_10m__v1_metrics_blockchain_utxo_created_value_median',
            'gn_10m__v1_metrics_blockchain_utxo_created_value_sum',
            'gn_10m__v1_metrics_blockchain_utxo_spent_count',
            'gn_10m__v1_metrics_blockchain_utxo_spent_value_mean',
            'gn_10m__v1_metrics_derivatives_futures_open_interest_perpetual_sum',
            'gn_10m__v1_metrics_derivatives_options_volume_daily_sum',
            'gn_10m__v1_metrics_derivatives_options_volume_put_call_ratio',
            'gn_10m__v1_metrics_transactions_transfers_volume_adjusted_sum',
            'gn_10m__v1_metrics_transactions_transfers_to_otc_desks_count',
            'gn_10m__v1_metrics_supply_minted',
            'gn_10m__v1_metrics_transactions_transfers_volume_to_otc_desks_sum',
            'gn_10m__v1_metrics_transactions_transfers_volume_to_miners_sum',
            'gn_10m__v1_metrics_transactions_transfers_from_otc_desks_count',
            'gn_10m__v1_metrics_transactions_transfers_volume_miners_to_exchanges',
            'gn_10m__v1_metrics_transactions_transfers_volume_miners_net',
            'gn_10m__v1_metrics_transactions_transfers_from_miners_count',
            'gn_10m__v1_metrics_transactions_transfers_volume_from_otc_desks_sum',
            'gn_10m__v1_metrics_transactions_transfers_volume_entity_adjusted_mean',
            'gn_10m__v1_metrics_transactions_entity_adjusted_count',
            'gn_10m__v1_metrics_transactions_transfers_volume_from_miners_sum',
            'gn_10m__v1_metrics_transactions_size_mean',
            'gn_10m__v1_metrics_transactions_size_sum',
            'gn_10m__v1_metrics_transactions_transfers_to_miners_count',
            'gn_10m__v1_metrics_transactions_transfers_volume_adjusted_mean',
            'gn_10m__v1_metrics_transactions_transfers_volume_adjusted_median',
            'gn_10m__v1_metrics_transactions_transfers_volume_entity_adjusted_sum',
            'gn_10m__v1_metrics_transactions_transfers_volume_entity_adjusted_median',
            'gn_10m__v1_metrics_addresses_min_32_count',
            'gn_10m__v1_metrics_mempool_txs_value_sum',
            'gn_10m__v1_metrics_supply_burned',
            'gn_10m__v1_metrics_fees_gas_limit_tx_mean',
            'gn_10m__v1_metrics_fees_gas_price_mean',
            'gn_10m__v1_metrics_fees_gas_price_median',
            'gn_10m__v1_metrics_fees_gas_used_mean',
            'gn_10m__v1_metrics_fees_gas_used_median',
            'gn_10m__v1_metrics_indicators_sol_more_10y',
            'gn_10m__v1_metrics_indicators_sol_7y_10y',
            'gn_10m__v1_metrics_indicators_sol_6m_12m',
            'gn_10m__v1_metrics_indicators_sol_5y_7y',
            'gn_10m__v1_metrics_indicators_sol_3y_5y',
            'gn_10m__v1_metrics_indicators_sol_3m_6m',
            'gn_10m__v1_metrics_indicators_sol_2y_3y',
            'gn_10m__v1_metrics_fees_gas_used_sum',
            'gn_10m__v1_metrics_indicators_sol_1w_1m',
            'gn_10m__v1_metrics_indicators_sol_1m_3m',
            'gn_10m__v1_metrics_indicators_sol_1h_24h',
            'gn_10m__v1_metrics_indicators_sol_1h',
            'gn_10m__v1_metrics_indicators_sol_1d_1w',
            'gn_10m__v1_metrics_fees_gas_limit_tx_median',
            'gn_10m__v1_metrics_distribution_balance_wbtc',
            'gn_10m__v1_metrics_mining_volume_mined_sum',
            'gn_10m__v1_metrics_distribution_balance_mtgox_trustee',
            'gn_10m__v1_metrics_mining_miners_unspent_supply',
            'gn_10m__v1_metrics_mining_miners_outflow_multiple',
            'gn_10m__v1_metrics_mempool_txs_value_distribution',
            'gn_10m__v1_metrics_mempool_txs_size_sum',
            'gn_10m__v1_metrics_mempool_txs_size_distribution',
            'gn_10m__v1_metrics_mempool_txs_count_sum',
            'gn_10m__v1_metrics_mempool_txs_count_distribution',
            'gn_10m__v1_metrics_mempool_fees_sum',
            'gn_10m__v1_metrics_mempool_fees_median_relative',
            'gn_10m__v1_metrics_mempool_fees_distribution',
            'gn_10m__v1_metrics_mempool_fees_average_relative',
            'gn_10m__v1_metrics_lightning_nodes_count',
            'gn_10m__v1_metrics_lightning_network_capacity_sum',
            'gn_10m__v1_metrics_lightning_channels_count',
            'gn_10m__v1_metrics_lightning_channel_size_median',
            'gn_10m__v1_metrics_lightning_channel_size_mean',
            'gn_10m__v1_metrics_distribution_balance_luna_foundation_guard',
            'gn_10m__v1_metrics_indicators_sol_1y_2y']

    def bot_loop_start(self, **kwargs) -> None:

        # Across Token Features: BTC 5m
        self.dataframe_btc_5m = self.dp.get_pair_dataframe(pair='BTC/USDT', timeframe='5m')
        self.dataframe_btc_5m = indicator_helpers.add_all_ta_informative(
            self.dataframe_btc_5m, suffix='_btc')

        # Across Token Features: BTC 1h
        self.dataframe_btc_1h = self.dp.get_pair_dataframe(pair='BTC/USDT', timeframe='1h')
        self.dataframe_btc_1h = indicator_helpers.add_all_ta_informative(
            self.dataframe_btc_1h, suffix='_btc')

        """# Across Token Features: BTC Glassnode
        gn_btc = []
        for f in self.gn_btc_f:
            SUFFIX = '_ppo'
            f_df = self.gn.query_metric(table_name=f, token='BTC',
                                        date_from='2021-01-01', date_to='2021-12-01',
                                        cols_to_drop=['token', 'update_timestamp'])
            f_df = indicator_helpers.add_single_ta_informative(
                f_df, ta.momentum.ppo, suffix=SUFFIX, col=f)
            f_df.set_index(keys='date' + SUFFIX, inplace=True)
            gn_btc.append(f_df)
        self.gn_btc_data = pd.concat(gn_btc, axis=1)
        self.gn_btc_data.reset_index(inplace=True)"""

    def populate_indicators(self, dataframe: DataFrame,
                            metadata: dict) -> DataFrame:

        # Apply TA indicators to the pair being traded
        dataframe = indicator_helpers.add_all_ta_informative(
            dataframe, suffix='')

        # Get glassnode signals & derive TA indicators
        token = metadata['pair'].split('/')[0]
        gn_data = defaultdict(dict)  # type: ignore
        for i, f in enumerate(self.gn_f):
            SUFFIX = '_ppo'
            gn_data[f]['df'] = self.gn.query_metric(table_name=f, token=token,
                                                    date_from='2021-01-01', date_to='2021-12-01',
                                                    cols_to_drop=['token', 'update_timestamp'])
            gn_data[f]['ta_df'] = indicator_helpers.add_single_ta_informative(
                gn_data[f]['df'], ta.momentum.ppo, suffix=SUFFIX, col=f)
            gn_data[f]['date_key'] = 'date' + SUFFIX

        # Add informative signals of pair at lower time resolution
        low_res_tf = '1h'
        dataframe_low_res = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe=low_res_tf)
        dataframe_low_res = indicator_helpers.add_all_ta_informative(
            dataframe_low_res, suffix='')

        # Add features/columns from imported litmus strategy
        dataframe = litmus.populate_indicators(dataframe)

        # Merge informative pairs
        # Note: For dataframes fetched in `bot_loop_start`, `.copy` is needed to avoid `date`
        # column being removed during merge which results in column
        # not available for joining for subsequent pairs

        # BTC price signals
        dataframe = merge_informative_pair(
            dataframe=dataframe, informative=self.dataframe_btc_5m.copy(), timeframe=self.timeframe,
            timeframe_inf='5m', ffill=True, date_column='date_btc')
        dataframe = merge_informative_pair(
            dataframe=dataframe, informative=self.dataframe_btc_1h.copy(), timeframe=self.timeframe,
            timeframe_inf='1h', ffill=True, date_column='date_btc')

        # Lower res pair price signals
        dataframe = merge_informative_pair(
            dataframe=dataframe, informative=dataframe_low_res, timeframe=self.timeframe,
            timeframe_inf='1h', ffill=True, date_column='date')

        # Glassnode: Per token features
        for f in gn_data.keys():
            dataframe = merge_informative_pair(
                dataframe=dataframe, informative=gn_data[f]['ta_df'], timeframe=self.timeframe,
                timeframe_inf='10m', ffill=True, date_column=gn_data[f]['date_key'])
            gn_data[f]['date_key'] += '_10m'
            dataframe.drop(columns=gn_data[f]['date_key'], inplace=True)

        """# Glassnode: Across token features (BTC)
        dataframe = merge_informative_pair(
            dataframe=dataframe, informative=self.gn_btc_data.copy(), timeframe=self.timeframe,
            timeframe_inf='10m', ffill=True, date_column='date_ppo')"""

        # Add reference to pair so ML model can generate feature for prediction
        dataframe['pair_copy'] = metadata['pair']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Buy trend indicator when buy_trigger crosses from 0 to 1
            ----------
            dataframe: DataFrame populated with indicators
            metadata: Additional information, like the currently traded pair
            return: DataFrame with buy column
        """

        dataframe = litmus.populate_entry_trend(dataframe)

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        """dataframe.loc[
            (qtpylib.crossed_below(dataframe['kama_exit_pos'], 0.5))
            & (dataframe['volume'] > 0),
            ['exit_long', 'enter_tag']] = (1, 'exit_long_trigger')"""

        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time, current_rate: float,
                    current_profit: float, **kwargs):
        """Custom exit needed to sell all trades open after X minutes & to label that forced
        sale as such. minimal_roi was not suitable for this because it would not allow
        labeling of the forced sell reason (ie. vertical barrier vs upper barrier hit).
            ----------
            pair : Pair that's currently analyzed
            trade : trade object
            current_time : datetime object, containing the current datetime
            current_rate : Current price, calculated based on pricing settings in ask_strategy
            current_profit : Current profit (as ratio), calculated based on current_rate
            return : custom sell_reason
        """

        # Sell any positions at a loss if they are held for more than X seconds.
        vertical_barrier_seconds = 6 * 60 * 60
        if (current_time - trade.open_date_utc).seconds > vertical_barrier_seconds:
            return 'vertical_barrier_force_sell'

        # Close trades when they exceed upper/lower barriers
        trade_open_rate = trade.open_rate
        PT_MULTIPLIER = 1.02
        SL_MULTIPLIER = 0.98

        # Profit taking upper barrier sell trigger
        if current_rate > trade_open_rate * PT_MULTIPLIER:
            return 'upper_barrier_pt_sell'

        # Stop loss lower barrier sell trigger
        elif current_rate < trade_open_rate * SL_MULTIPLIER:
            return 'lower_barrier_sl_sell'

        return None
