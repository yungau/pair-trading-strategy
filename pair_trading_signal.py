#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 22:14:04 2023

@author: chingyungau
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.float_format', '{:_.4f}'.format)
plt.rcParams['figure.figsize'] = (16, 8)


class SignalGeneration:

    def __init__(self,
        n_rolling=63, n_std_enter=2, n_std_exit=1,
        take_log=True, max_holding_days=25, n_day_lag=1,
        use_hedge_ratio=True,
        ticker1=None, ticker2=None,
        **params):

        self.n_rolling = n_rolling
        self.n_std_enter = n_std_enter
        self.n_std_exit = n_std_exit
        self.take_log = take_log
        self.max_holding_days = max_holding_days
        self.n_day_lag = n_day_lag
        self.use_hedge_ratio = use_hedge_ratio
        self.px_ts1 = None
        self.px_ts2 = None
        self.params = params
        self.n_date_drop = self.params.get('n_date_drop', 20)
        self.model_output = None
        self.param_str = ('log' if take_log else '') + ' ' + \
                          f'{self.n_rolling}_{self.n_std_enter}_{self.n_std_exit}'
        self.param_str = self.param_str.strip()
        self.ticker1 = ticker1
        self.ticker2 = ticker2

    @staticmethod
    def cal_hedge_ratio(px_ts1, px_ts2, **params):
        trans_mat_T = np.eye(2)
        # observation matrix F is 2-dimensional, containing 1 and price2 series
        # there are px_ts2.shape[0] observations
        obs_mat_F = np.transpose(
            np.vstack([
                np.ones(px_ts2.shape[0]),
                px_ts2.values
            ])
        ).reshape(-1, 1, 2)

        delta = params.get('delta', 0.0001)
        trans_cov = delta / (1 - delta) * np.eye(2)  # Q

        kf = KalmanFilter(
            n_dim_obs=params.get('n_dim_obs', 1),
            n_dim_state=params.get('n_dim_state', 2),
            transition_matrices=params.get('transition_matrices', trans_mat_T),
            observation_matrices=params.get('observation_matrices', obs_mat_F),
            initial_state_mean=params.get('initial_state_mean', np.zeros(2)),
            initial_state_covariance=params.get(
                'initial_state_covariance', np.ones((2, 2))),
            observation_covariance=params.get('observation_covariance', 0.5),
            transition_covariance=params.get(
                'transition_covariance', trans_cov),
        )

        state_means, state_covs = kf.filter(px_ts1)

        output = {
            'kf': kf,
            'state_means': state_means,
            'state_covs': state_covs,
            'spread_mean': state_means[:, 0],
            'hedge_ratio': state_means[:, 1],
        }

        return output

    def cal_spread_ts(self, df):
        assert set(['px_ts1', 'px_ts2', 'spread_mean',
                   'hedge_ratio']) <= set(df.columns)

        df['est_px_ts1'] = df['px_ts2'] * df['hedge_ratio'] + df['spread_mean']
        df['demeaned_spread'] = df['px_ts1'] - df['est_px_ts1']
        self.df = df
        return df['demeaned_spread'].copy()

    def cal_zscore_signal_df(self, spread_ts):

        n_rolling = self.n_rolling
        n_std_enter = self.n_std_enter
        n_std_exit = self.n_std_exit

        df = self.df
        df['demeaned_spread'] = spread_ts
        df['std_rolling'] = df['demeaned_spread'].rolling(n_rolling).std()
        df['lower_band_enter'] = -n_std_enter * df['std_rolling']
        df['upper_band_enter'] = n_std_enter * df['std_rolling']
        df['lower_band_exit'] = -n_std_exit * df['std_rolling']
        df['upper_band_exit'] = n_std_exit * df['std_rolling']

        df['below_lower_band_enter'] = np.where(
            df['demeaned_spread'] < df['lower_band_enter'], 1, 0)
        df['below_lower_band_exit'] = np.where(
            df['demeaned_spread'] < df['lower_band_exit'], 1, 0)
        df['above_upper_band_enter'] = np.where(
            df['demeaned_spread'] > df['upper_band_enter'], 1, 0)
        df['above_upper_band_exit'] = np.where(
            df['demeaned_spread'] > df['upper_band_exit'], 1, 0)

        df['undervalued_start'] = np.where(
            (df['below_lower_band_enter'] == 1) &
            (df['below_lower_band_enter'].shift(1) == 0),
            1, 0)
        df['undervalued_end'] = np.where(
            (df['below_lower_band_exit'] == 0) &
            (df['below_lower_band_exit'].shift(1) == 1),
            1, 0)
        df['overvalued_start'] = np.where(
            (df['above_upper_band_enter'] == 1) &
            (df['above_upper_band_enter'].shift(1) == 0),
            1, 0)
        df['overvalued_end'] = np.where(
            (df['above_upper_band_exit'] == 0) &
            (df['above_upper_band_exit'].shift(1) == 1),
            1, 0)

        df['undervalued'] = ((df['undervalued_start'] - df['undervalued_end'])
                            .replace(0, np.nan)
                            .fillna(method='ffill')
                            .replace(-1, 0)
                            .fillna(0))
        df['overvalued'] = ((df['overvalued_start'] - df['overvalued_end'])
                            .replace(0, np.nan)
                            .fillna(method='ffill')
                            .replace(-1, 0)
                            .fillna(0))

        # signal = 1 <=> spread is overvalued
        # signal = -1 <=> spread is undervalued
        df['temp_signal'] = df['overvalued'] - df['undervalued']

        return df

    @staticmethod
    def count_holding_days_df(ts):
        ts = ts.copy()
        df = pd.DataFrame({'signal': ts})
        prev_signal = np.nan
        start_date = df.iloc[0].index
        for date, row in df.iterrows():
            if (row['signal'] == 0) or (row['signal'] != prev_signal):
                start_date = date  # reset start_date
                df.loc[date, 'start_date'] = start_date

            df.loc[date, 'start_date'] = start_date
            prev_signal = row['signal']
        df['holding_days'] = (df.index - df['start_date']).dt.days + 1
        return df

    @staticmethod
    def cal_weights_df(signals_df, use_hedge_ratio):
        assert set(['px_ts1', 'px_ts2', 'spread_mean',
                   'hedge_ratio']) <= set(signals_df.columns)
        signals_df = signals_df.copy()

        signals_df = signals_df.copy()
        # signal = 1 <--> ticker1 overvalued or ticker2 undervalued
        signals_df['signal1'] = -signals_df['signal']
        signals_df['signal2'] = -signals_df['signal1']
        signals_df['weight'] = 1
        if use_hedge_ratio:
            signals_df['weight2'] = signals_df['hedge_ratio'] * \
                signals_df['px_ts2'] / signals_df['px_ts1']
        else:
            signals_df['weight2'] = 1
        return signals_df

    def run(self, px_ts1, px_ts2):

        self.px_ts1 = px_ts1
        self.px_ts2 = px_ts2

        tmp_df = pd.concat([px_ts1, px_ts2], axis=1, sort=False).fillna(
            method='ffill').dropna()
        px_ts1, px_ts2 = tmp_df.iloc[:, 0], tmp_df.iloc[:, 1]

        if self.take_log:
            px_ts1, px_ts2 = np.log(px_ts1), np.log(px_ts2)

        model_output = self.cal_hedge_ratio(px_ts1, px_ts2, **self.params)
        self.model_output = model_output
        df = pd.DataFrame({
            'px_ts1': px_ts1,
            'px_ts2': px_ts2,
            'spread_mean': model_output['spread_mean'],
            'hedge_ratio': model_output['hedge_ratio'],
            })
        df = df.iloc[self.n_date_drop:, :]

        spread_ts = self.cal_spread_ts(df)
        signals_df = self.cal_zscore_signal_df(spread_ts)

        # shift the signal date by n_day_lag
        signals_df = signals_df.shift(self.n_day_lag).iloc[self.n_day_lag:, :]
        signals_df['holding_days'] = self.count_holding_days_df(
            signals_df['temp_signal'])['holding_days']
        if self.max_holding_days is not None:
            # force unwind if the number of holding days > max_holding_days
            signals_df['signal'] = np.where(
                signals_df['holding_days'] <= self.max_holding_days, signals_df['temp_signal'], 0)
        else:
            signals_df['signal'] = signals_df['temp_signal']

        # use_hedge_ratio is False, equal weight on ticker1 and ticker2
        signals_df = self.cal_weights_df(signals_df, self.use_hedge_ratio)

        self.get_tickers()

        self.df = signals_df

        return signals_df

    def get_tickers(self):
        if self.ticker1 is None:
            self.ticker1 = self.px_ts1.name if self.px_ts1 is not None else 'px_ts1'
        if self.ticker2 is None:
            self.ticker2 = self.px_ts2.name if self.px_ts2 is not None else 'px_ts2'

    def plot_intermediate_results(self, show_days=100, marker='o--'):
        
        if show_days == 'all':
            show_df = self.df
            marker = '-'
        else:
            show_df = self.df.tail(show_days)

        fig, ax = plt.subplots(nrows=2, sharex=True)
        ax[0].plot(show_df['px_ts1'], label=('log ' + self.ticker1).strip())
        ax[1].plot(show_df['px_ts2'], label=('log ' + self.ticker2).strip())
        ax[0].legend()
        ax[0].grid()
        ax[1].legend()
        ax[1].grid()
        plt.show()

        plt.plot(show_df['demeaned_spread'], marker, label='spread')
        plt.plot(show_df['upper_band_enter'], ls='--', label='upper_band_enter')
        plt.plot(show_df['lower_band_enter'], ls='--', label='lower_band_enter')
        plt.plot(show_df['upper_band_exit'], ls='--', label='upper_band_exit')
        plt.plot(show_df['lower_band_exit'], ls='--', label='lower_band_exit')
        plt.title(self.param_str)
        plt.legend()
        plt.grid()
        plt.show()
        
        fig, ax=plt.subplots(nrows=3, sharex=True)
        ax[0].plot(show_df['spread_mean'], marker, label='spread_mean')
        ax[1].plot(show_df['hedge_ratio'], marker, label='hedge_raio')
        ax[2].plot(show_df['hedge_ratio'], marker,
                   label='weight on ' + self.ticker2)
        ax[0].legend()
        ax[0].grid()
        ax[1].legend()
        ax[1].grid()
        ax[2].legend()
        ax[2].grid()
        plt.show()

        _ = show_df[['px_ts1', 'est_px_ts1']].plot(grid=True)
        plt.show()

        _ = (show_df['demeaned_spread'] / show_df['px_ts1'] * 100).plot(
                title='% error of estimation of px_ts1 (spread / px_ts1 * 100)' )
        plt.show()

        
