"""
FreqAI-kompatible jebana Strategie
Konvertiert von der ursprünglichen jebana.py Strategie
"""

from datetime import datetime
import pandas as pd
import numpy as np
from freqtrade.persistence import Trade
from freqtrade.strategy import DecimalParameter, IntParameter, CategoricalParameter, merge_informative_pair
from functools import lru_cache, reduce
import os
import logging
from typing import Optional, Dict, Any
from pandas import DataFrame, Series

# FreqAI Imports
from freqtrade.strategy import IStrategy

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
log = logging.getLogger(__name__)


class jebanaCL(IStrategy):
    _columns_to_expect = [
        '&s-gain_jebana'
    ]

    # Hyperoptbare Parameter (aus der ursprünglichen jebana)
    ts_n_profit_std = DecimalParameter(0.0, 3.0, decimals=2, default=2.0, space="buy")
    ts_n_loss_std = DecimalParameter(0.0, 3.0, decimals=2, default=1.0, space="buy")
    
    # Chandelier-Parameter
    ce_len = IntParameter(low=10, high=50, default=22, space='sell')
    ce_mult = DecimalParameter(default=3.0, low=1.5, high=4.0, decimals=2, space='sell')
    rr_target = DecimalParameter(default=2.0, low=0.5, high=4.0, decimals=2, space='sell')
    
    # Leverage
    leverage_amount = IntParameter(low=1, high=10, default=3, space="buy")
    
    # Guard Conditions (aus NNPredict)
    enable_guard_metric = CategoricalParameter([True, False], default=False, space="buy")
    enable_bb_check = CategoricalParameter([True, False], default=False, space="buy")
    enable_squeeze = CategoricalParameter([True, False], default=False, space="buy")
    
    entry_guard_metric = DecimalParameter(-0.8, -0.2, default=-0.2, decimals=1, space="buy")
    entry_bb_width = DecimalParameter(0.020, 0.100, default=0.02, decimals=3, space="buy")
    entry_bb_factor = DecimalParameter(0.70, 1.20, default=1.1, decimals=2, space="buy")

    win_size = 14
    
    # Mindestschranken analog NNPredict
    cexit_min_profit_th = DecimalParameter(0.0, 1.5, default=0.7, decimals=1, space="buy")
    cexit_min_loss_th = DecimalParameter(-1.5, -0.0, default=-0.4, decimals=1, space="sell")
    
    # Strategy Parameter
    minimal_roi = {"0": 0.1}
    stoploss = -0.99
    trailing_stop = False
    use_custom_stoploss = True
    can_short = True
    timeframe = "5m"
    startup_candle_count = 600
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], 
                 side: str, **kwargs) -> float:
        """Leverage Berechnung"""
        return self.leverage_amount.value

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        # Get data from producer
        pair = metadata['pair']
        timeframe = self.timeframe

        # Process producer
        producer, _ = self.dp.get_producer_df(pair, timeframe=timeframe, producer_name="jebana_origin")
        if not producer.empty:
            dataframe = merge_informative_pair(dataframe, producer,
                                          timeframe, timeframe,
                                          append_timeframe=False,
                                          suffix="jebana")
        else:
            dataframe[self._columns_to_expect] = 0
        
        # Bollinger Bands (must include these)
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['%%-bb_lowerband'] = bollinger['lower']
        dataframe['%%-bb_middleband'] = bollinger['mid']
        dataframe['%%-bb_upperband'] = bollinger['upper']
        dataframe['%%-bb_width'] = ((dataframe['%%-bb_upperband'] - dataframe['%%-bb_lowerband']) / dataframe['%%-bb_middleband'])
        dataframe["%%-bb_gain"] = ((dataframe["%%-bb_upperband"] - dataframe["close"]) / dataframe["close"])
        dataframe["%%-bb_loss"] = ((dataframe["%%-bb_lowerband"] - dataframe["close"]) / dataframe["close"])


        # RMI: https://www.tradingview.com/script/kwIt9OgQ-Relative-Momentum-Index/
        rmi = RMI(dataframe, length=self.win_size, mom=5)

        # scaled version for use as guard metric
        srmi = 2.0 * (rmi - 50.0) / 100.0

        # guard metric must be in range [-1,+1], with -ve values indicating oversold and +ve values overbought
        dataframe["%%-guard_metric"] = srmi

        return dataframe
    
    def populate_entry_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        
        # ATR (True Range, EMA-glättung)
        tr = pd.concat([
            (df['high'] - df['low']).abs(),
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        df['atr'] = tr.ewm(alpha=1/(int(self.ce_len.value)), adjust=False).mean()
        
        # HH/LL ohne Lookahead
        df['hh'] = df['high'].rolling(int(self.ce_len.value), min_periods=int(self.ce_len.value)).max().shift(1)
        df['ll'] = df['low'].rolling(int(self.ce_len.value), min_periods=int(self.ce_len.value)).min().shift(1)
        
        # Bänder
        df['ce_long'] = df['hh'] - df['atr'] * float(self.ce_mult.value)
        df['ce_short'] = df['ll'] + df['atr'] * float(self.ce_mult.value)
        
        """Entry-Signale basierend auf FreqAI Vorhersagen mit Guard-Conditions"""
        
        # Initialisiere Spalten
        df.loc[:, "enter_tag"] = ""
        df["enter_long"] = 0
        df["buy_region"] = 0
        df["curr_target"] = 0.0
        
        # FreqAI Vorhersage-Spalte (wird automatisch von FreqAI erstellt)
        prediction_col = '&s-gain_jebana'

        # Target-Berechnung auf Basis realisierter Gewinne (analog NNPredict)
        profit_nstd = float(self.ts_n_profit_std.value)
        loss_nstd   = float(self.ts_n_loss_std.value)
        win_size    = 12

        # Realisierte Gewinne aus Preisreihe (Trailing über Label-Periode)
        lookback = 12
        realized_gain = 100.0 * (df["close"] - df["close"].shift(lookback)) / df["close"].shift(lookback)
        realized_gain = realized_gain.fillna(0.0)
        realized_profit = realized_gain.clip(lower=0.0)
        realized_loss   = realized_gain.clip(upper=0.0)

        df["target_profit"] = (
            realized_profit.rolling(win_size, min_periods=win_size).mean()
            + profit_nstd * realized_profit.rolling(win_size, min_periods=win_size).std(ddof=0)
        )

        df["target_loss"] = (
            realized_loss.rolling(win_size, min_periods=win_size).mean()
            - loss_nstd * realized_loss.rolling(win_size, min_periods=win_size).std(ddof=0).abs()
        )

        # Mindestschranken übernehmen (wie NNPredict)
        df["target_profit"] = df["target_profit"].clip(lower=float(self.cexit_min_profit_th.value))
        df["target_loss"] = df["target_loss"].clip(upper=float(self.cexit_min_loss_th.value))
        
        # Entry-Bedingungen
        conditions = []
        
        # Volumen-Check
        conditions.append(df["volume"] > 1.0)
        
        # Guard-Conditions (aus NNPredict)
        guard_conditions = []

        if self.enable_guard_metric.value:
            # Guard metric in oversold region
            guard_conditions.append(df["%%-guard_metric"] < self.entry_guard_metric.value)

        if self.enable_bb_check.value:
            # Bollinger band-based bull/bear indicators
            lower_limit = df["%%-bb_middleband"] - self.entry_bb_factor.value * (
                df["%%-bb_middleband"] - df["%%-bb_lowerband"]
            )
            df["bullish"] = np.where((df["close"] <= lower_limit), 1, 0)
            guard_conditions.append(df["bullish"] > 0)

        if self.enable_squeeze.value:
            if "squeeze" not in df.columns:
                df["squeeze"] = np.where((df["%%-bb_width"] >= self.entry_bb_width.value), 1, 0)
            guard_conditions.append(df["squeeze"] > 0)

        # Buy region kombinieren (für Plotting)
        if guard_conditions:
            df.loc[reduce(lambda x, y: x & y, guard_conditions), "buy_region"] = 1

            # Model triggers mit Guard-Conditions
            model_cond = (
                # buy region
                (df["buy_region"] > 0)
                & (
                    # prediction crossed target
                    qtpylib.crossed_above(df[prediction_col], df["target_profit"])
                )
            )
        else:
            # Model triggers ohne Guard-Conditions
            model_cond = (
                # prediction crossed target
                qtpylib.crossed_above(df[prediction_col], df["target_profit"])
            )

        conditions.append(model_cond)
        
        # Chandelier Exit Long (für Long-Entry)
        # conditions.append(df["close"] > df["ce_long"])
        
        # Entry-Signal setzen
        if conditions:
            df.loc[reduce(lambda x, y: x & y, conditions), "enter_long"] = 1

        # Entry-Tags setzen
        df.loc[model_cond, "enter_tag"] += "model_entry "

        # Target für aktuellen Trade setzen
        curr_target = 0.0
        for i in range(len(df["close"])):
            if df["enter_long"].iloc[i] > 0.0:
                curr_target = df["close"].iloc[i] * (1.0 + df[prediction_col].iloc[i] / 100.0)
            df.loc[i, "curr_target"] = curr_target
        
        return df

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> bool:
        # in Backtest/Plot/Hyperopt immer zulassen
        if self.dp.runmode.value in ("backtest", "plot", "hyperopt"):
            return True

        # Gegen aktuelles Ziel prüfen
        df, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if df is None or df.empty or ("curr_target" not in df.columns):
            return True

        last = df.iloc[-1]
        curr_target = float(last.get("curr_target", 0.0))
        if curr_target > 0.0 and rate >= curr_target:
            print("")
            print(f"    *** {pair} Trade abgelehnt. Rate ({rate:.2f}) über Ziel ({curr_target:.2f}) ")
            print("")
            return False

        return True
    
    def populate_exit_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # """Exit-Signale basierend auf FreqAI Vorhersagen"""
        
        # prediction_col = '&s-gain'
        
        # # Exit-Bedingungen
        # conditions = []
        
        # # FreqAI Vorhersage unter Target Loss
        # conditions.append(df[prediction_col] < df["target_loss"])
        
        # # Chandelier Exit Short (für Long-Exit)
        # conditions.append(df["close"] < df["ce_short"])
        
        # # Exit-Signal setzen
        # df.loc[reduce(lambda x, y: x & y, conditions), "exit_long"] = 1
        
        return df
    
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """Custom Stoploss basierend auf Chandelier Exit"""
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        row = df.loc[df['date'] <= current_time].iloc[-1]
        
        if trade.is_short:
            sl_price = min(float(row['ce_short']), float(trade.stop_loss) if trade.stop_loss else float('inf'))
            dist = max(0.0, (sl_price - current_rate) / current_rate)
        else:
            sl_price = max(float(row['ce_long']), float(trade.stop_loss) if trade.stop_loss else 0.0)
            dist = max(0.0, (current_rate - sl_price) / current_rate)
        
        return dist
    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs):
        """Custom Exit basierend auf Risk:Reward Ratio"""
        
        # R:R - initiales Risiko aus ATR zur Entry-Kerze
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        entry_row = df.loc[df['date'] >= trade.open_date_utc].head(1)
        if entry_row.empty:
            return None
        
        atr0 = float(entry_row['atr'].iloc[0])
        risk_pct = (atr0 * float(self.ce_mult.value)) / float(trade.open_rate)
        
        if current_profit >= float(self.rr_target.value) * risk_pct and current_profit > 0:
            return "rr_tp"
        
        return None

def RMI(dataframe, *, length=20, mom=5):
    """
    Source: https://github.com/freqtrade/technical/blob/master/technical/indicators/indicators.py#L912
    """
    df = dataframe.copy()

    df['maxup'] = (df['close'] - df['close'].shift(mom)).clip(lower=0)
    df['maxdown'] = (df['close'].shift(mom) - df['close']).clip(lower=0)

    df.fillna(0, inplace=True)

    df["emaInc"] = ta.EMA(df, price='maxup', timeperiod=length)
    df["emaDec"] = ta.EMA(df, price='maxdown', timeperiod=length)

    df['RMI'] = np.where(df['emaDec'] == 0, 0, 100 - 100 / (1 + df["emaInc"] / df["emaDec"]))

    return df["RMI"]