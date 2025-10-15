"""
FreqAI-kompatible jebana Strategie
Konvertiert von der ursprünglichen jebana.py Strategie
"""

from datetime import datetime
import pandas as pd
import numpy as np
from freqtrade.persistence import Trade
from freqtrade.strategy import (
    DecimalParameter,
    IntParameter,
    CategoricalParameter,
    stoploss_from_absolute,
    timeframe_to_minutes,
    timeframe_to_prev_date,
    stoploss_from_open,
    Order,
)
from functools import lru_cache, reduce
import os
import logging
from typing import Optional, Dict, Any, Union
from pandas import DataFrame, Series

# FreqAI Imports
from freqtrade.strategy import IStrategy
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen

import talib.abstract as ta
from scipy.signal import argrelextrema
import freqtrade.vendor.qtpylib.indicators as qtpylib
from finta import TA as fta
import pywt
import math
import re
from datetime import timedelta

log = logging.getLogger(__name__)


class jebana_freqai(IStrategy):
    # Hyperoptbare Parameter (aus der ursprünglichen jebana)
    ts_n_profit_std = DecimalParameter(0.0, 3.0, decimals=2, default=2.0, space="buy")
    ts_n_loss_std = DecimalParameter(0.0, 3.0, decimals=2, default=1.0, space="buy")

    # Chandelier-Parameter
    ce_len = IntParameter(low=10, high=50, default=22, space="sell")
    ce_mult = DecimalParameter(default=3.0, low=1.5, high=4.0, decimals=2, space="sell")
    rr_target = DecimalParameter(default=2.0, low=0.5, high=4.0, decimals=2, space="sell")

    # Microstructure Parameter
    use_micro = CategoricalParameter([True, False], default=True, space="buy")
    micro_path = "/home/llm/datastream-3/crypto_data"
    trades_win = IntParameter(6, 96, default=24, space="buy")
    trades_z_win = IntParameter(50, 600, default=288, space="buy")
    trades_ema = IntParameter(3, 60, default=12, space="buy")

    oi_win = IntParameter(6, 96, default=24, space="buy")
    oi_z_win = IntParameter(50, 600, default=288, space="buy")
    oi_ema = IntParameter(3, 60, default=12, space="buy")

    # Liquidation Parameter
    use_liq = CategoricalParameter([True, False], default=True, space="buy")
    liq_path = "/home/llm/datastream-3/crypto_data"
    liq_z_win = IntParameter(50, 600, default=288, space="buy")

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

    # Spaltenmapping
    col_date = "date"
    col_trades = "trades_count"
    col_oi = "oi_avg"

    # Strategy Parameter
    minimal_roi = {"0": 0.1}
    stoploss = -0.99
    trailing_stop = False
    use_custom_stoploss = True
    can_short = True
    timeframe = "5m"
    startup_candle_count = 600
    timeframe_minutes = timeframe_to_minutes(timeframe)

    plot_config = {
        "main_plot": {"ce_long": {"color": "#ff0000", "type": "line"}},
        "subplots": {
            "predictions": {
                "&s-gain": {"color": "#ffffff", "type": "line"},
                "target_profit": {"color": "#00ff66", "type": "line"},
                "target_loss": {"color": "#ff0000", "type": "line"},
            },
            "guards": {"%%-guard_metric": {"color": "#7dc990"}},
            "bullish": {"bullish": {"color": "#0b582c"}},
            "squeeze": {"squeeze": {"color": "#34bc0b"}},
        },
    }

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        """Leverage Berechnung"""
        return self.leverage_amount.value

    def _norm_pair(self, pair: str) -> str:
        """Pair normalisieren: ADA/USDT:USDT -> ADA-USDT"""
        return pair.split(":")[0].replace("/", "-")

    def _tf_to_pandas(self, tf: str) -> str:
        """Timeframe konvertieren: 5m -> 5min"""
        return tf[:-1] + "min" if tf.endswith("m") else tf

    def _load_micro(self, pair: str) -> pd.DataFrame:
        """Lädt Microstructure Daten"""
        sym = self._norm_pair(pair)
        base = os.path.join(self.micro_path, f"{sym}_combined")
        df = None

        for ext in [".csv"]:
            p = base + ext
            if os.path.exists(p):
                df = pd.read_csv(p)
                break

        if df is None:
            raise FileNotFoundError(f"Microstructure file not found for {sym} in {self.micro_path}")

        # Zeitstempel konvertieren
        df[self.col_date] = pd.to_datetime(df[self.col_date], utc=True)

        # Spalten existieren?
        if self.col_trades not in df.columns:
            for alt in ["trades", "count", "side_buy", "liq_count"]:
                if alt in df.columns:
                    self.col_trades = alt
                    break
        if self.col_oi not in df.columns:
            for alt in ["oi_avg", "openInterest", "open_interest_value"]:
                if alt in df.columns:
                    self.col_oi = alt
                    break

        # Minimal set behalten
        cols = [self.col_date]
        if self.col_trades in df.columns:
            cols.append(self.col_trades)
        if self.col_oi in df.columns:
            cols.append(self.col_oi)
        df = df[cols].copy().sort_values(self.col_date)

        # Numerisch und >=0 sicherstellen
        for c in cols[1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(lower=0.0)

        return df

    def _prep_micro(self, pair: str, timeframe: str) -> pd.DataFrame:
        """Bereitet Microstructure Daten vor"""
        df = self._load_micro(pair)
        rule = self._tf_to_pandas(timeframe)

        # Resample auf Strategy-Timeframe
        out = (
            df.set_index(self.col_date)
            .resample(rule)
            .agg(
                {
                    self.col_trades: "sum" if self.col_trades in df.columns else "sum",
                    self.col_oi: "last" if self.col_oi in df.columns else "last",
                }
            )
            .rename(columns={self.col_trades: "trades_count", self.col_oi: "oi"})
            .reset_index()
            .rename(columns={self.col_date: "date"})
        )

        # Rolling Features
        tw = int(self.trades_win.value)
        tz = int(self.trades_z_win.value)
        emaT = int(self.trades_ema.value)

        ow = int(self.oi_win.value)
        oz = int(self.oi_z_win.value)
        emaO = int(self.oi_ema.value)

        if "trades_count" in out.columns:
            out["%-trades_sum"] = out["trades_count"].rolling(tw, min_periods=max(3, tw // 3)).sum()
            out["%-trades_mean"] = (
                out["trades_count"].rolling(tw, min_periods=max(3, tw // 3)).mean()
            )
            out["%-trades_ema"] = out["trades_count"].ewm(span=emaT, adjust=False).mean()
            out["%-trades_momo"] = (
                out["%-trades_ema"] / out["%-trades_ema"].shift(emaT) - 1.0
            ).fillna(0.0)

            # Z-Score
            rm = out["trades_count"].rolling(tz, min_periods=max(10, tz // 5)).mean()
            rs = out["trades_count"].rolling(tz, min_periods=max(10, tz // 5)).std()
            eps = 1e-9
            z = (out["trades_count"] - rm) / rs.replace(0, eps)
            out["%-trades_z"] = z.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-10, 10)

        if "oi" in out.columns:
            out["%-oi"] = out["oi"].ffill().fillna(0.0)
            out["%-oi_delta"] = out["%-oi"].diff().fillna(0.0)
            out["%-oi_pct"] = (
                (out["%-oi"] / out["%-oi"].shift() - 1.0)
                .replace([pd.NA, pd.NaT], 0)
                .fillna(0.0)
                .clip(-1, 1)
            )
            out["%-oi_ema"] = out["%-oi"].ewm(span=emaO, adjust=False).mean()
            out["%-oi_momo"] = (out["%-oi_ema"] / out["%-oi_ema"].shift(emaO) - 1.0).fillna(0.0)

            # Z-Score
            rm = out["%-oi"].rolling(oz, min_periods=max(10, oz // 5)).mean()
            rs = out["%-oi"].rolling(oz, min_periods=max(10, oz // 5)).std()
            eps = 1e-9
            z = (out["%-oi"] - rm) / rs.replace(0, eps)
            out["%-oi_z"] = z.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-10, 10)

        return out

    def _merge_micro(self, df: pd.DataFrame, pair: str) -> pd.DataFrame:
        """Merge Microstructure Daten"""
        micro = self._prep_micro(pair, self.timeframe)
        if self.timeframe in ("5m", "5T"):
            out = df.merge(micro, on="date", how="left")
        else:
            out = pd.merge_asof(
                df.sort_values("date"),
                micro.sort_values("date"),
                on="date",
                direction="backward",
                tolerance=pd.Timedelta("10m"),
            )

        fillcols = [
            c
            for c in [
                "trades_count",
                "%-trades_sum",
                "%-trades_mean",
                "%-trades_ema",
                "%-trades_momo",
                "%-trades_z",
                "%-oi",
                "%-oi_delta",
                "%-oi_pct",
                "%-oi_ema",
                "%-oi_momo",
                "%-oi_z",
            ]
            if c in out.columns
        ]
        out[fillcols] = out[fillcols].fillna(0.0)
        return out

    def _load_liq_df(self, pair: str) -> pd.DataFrame:
        """Lädt Liquidation Daten"""
        sym = pair.split(":")[0].replace("/", "-")
        base = os.path.join(self.liq_path, f"{sym}_combined")
        df = None

        for ext in [".csv"]:
            p = base + ext
            if os.path.exists(p):
                df = pd.read_csv(p)
                break

        if df is None:
            alt = os.path.join(self.liq_path, f"{sym}_combined.csv")
            df = pd.read_csv(alt)

        df["date"] = pd.to_datetime(df["date"], utc=True)

        # Grund-Features
        df["side_buy"] = df["side_buy"].fillna(0.0)
        df["side_sell"] = df["side_sell"].fillna(0.0)
        df["%-liq_tot"] = df["side_buy"] + df["side_sell"]
        price_col = "price_avg" if "price_avg" in df.columns else "avg_price"
        df["%-liq_notional"] = df["total_quantity"].fillna(0.0) * df[price_col].fillna(0.0)

        # Z-Score
        zN = int(self.liq_z_win.value)
        roll = df["%-liq_tot"].rolling(zN, min_periods=max(10, zN // 5))
        df["%-liq_z"] = ((df["%-liq_tot"] - roll.mean()) / roll.std()).fillna(0.0).clip(-10, 10)
        df["%-liq_buy_ratio"] = (
            df["side_buy"] / df["%-liq_tot"].where(df["%-liq_tot"] > 0, 1)
        ).fillna(0.0)
        df = df.sort_values("date").reset_index(drop=True)

        return df[
            [
                "date",
                "%-liq_tot",
                "%-liq_notional",
                "%-liq_z",
                "%-liq_buy_ratio",
                "side_buy",
                "side_sell",
            ]
        ]

    def _merge_liq(self, df: pd.DataFrame, pair: str) -> pd.DataFrame:
        """Merge Liquidation Daten"""
        liq = self._load_liq_df(pair)

        if self.timeframe in ("5m", "5T"):
            out = df.merge(liq, on="date", how="left")
        else:
            out = pd.merge_asof(
                df.sort_values("date"),
                liq.sort_values("date"),
                on="date",
                direction="backward",
                tolerance=pd.Timedelta("10m"),
            )

        fillcols = [
            "%-liq_tot",
            "%-liq_notional",
            "%-liq_z",
            "%-liq_buy_ratio",
            "side_buy",
            "side_sell",
        ]
        out[fillcols] = out[fillcols].fillna(0.0)
        return out

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        dataframe["%-mid"] = (dataframe["open"] + dataframe["close"]) / 2.0

        lookahead = self.freqai_info["feature_parameters"]["label_period_candles"]

        bgain = (
            100.0
            * (dataframe["close"] - dataframe["close"].shift(lookahead))
            / dataframe["close"].shift(lookahead)
        )

        # Index-aligned, länge bleibt konstant
        s = pd.Series(np.asarray(bgain), index=dataframe.index)
        dataframe["%-gain"] = s.rolling(window=4, min_periods=1).mean().round(2)

        dataframe["%-profit"] = dataframe["%-gain"].clip(lower=0.0)
        dataframe["%-loss"] = dataframe["%-gain"].clip(upper=0.0)

        # RSI
        rsi = ta.RSI(dataframe, timeperiod=self.win_size)
        dataframe["%-rsi"] = rsi

        # Williams %R
        wr = 0.02 * (self.williams_r(dataframe, period=self.win_size) + 50.0)

        # Fisher RSI
        rsi_scaled = 0.1 * (rsi - 50)
        fisher_rsi = (np.exp(2 * rsi_scaled) - 1) / (np.exp(2 * rsi_scaled) + 1)

        # Combined Fisher RSI and Williams %R
        dataframe["%-fisher_wr"] = (wr + fisher_rsi) / 2.0

        # RMI: https://www.tradingview.com/script/kwIt9OgQ-Relative-Momentum-Index/
        rmi = RMI(dataframe, length=self.win_size, mom=5)

        # scaled version for use as guard metric
        srmi = 2.0 * (rmi - 50.0) / 100.0

        # guard metric must be in range [-1,+1], with -ve values indicating oversold and +ve values overbought
        dataframe["%%-guard_metric"] = srmi

        # MFI - Chaikin Money Flow Indicator
        dataframe["%-mfi"] = ta.MFI(dataframe)

        # Recent min/max
        dataframe["%-recent_min"] = dataframe["close"].rolling(window=self.win_size).min()
        dataframe["%-recent_max"] = dataframe["close"].rolling(window=self.win_size).max()

        # MACD
        macd = ta.MACD(dataframe)
        dataframe["%-macd"] = macd["macd"]
        dataframe["%-macdsignal"] = macd["macdsignal"]
        dataframe["%-macdhist"] = macd["macdhist"]

        # Stoch fast
        stoch_fast = ta.STOCHF(dataframe)
        dataframe["%-fastd"] = stoch_fast["fastd"]
        dataframe["%-fastk"] = stoch_fast["fastk"]
        dataframe["%-fast_diff"] = dataframe["%-fastd"] - dataframe["%-fastk"]

        # DWT model
        # if in backtest or hyperopt, then we have to do rolling calculations
        if hasattr(self, "runmode") and self.runmode in ("hyperopt", "backtest", "plot"):
            dataframe["%-dwt"] = (
                dataframe["%-mid"]
                .rolling(window=self.startup_candle_count)
                .apply(self.roll_get_dwt)
            )
        else:
            dataframe["%-dwt"] = self.get_dwt(dataframe["%-mid"])

        dataframe["%-dwt_gain"] = (
            100.0 * (dataframe["%-dwt"] - dataframe["%-dwt"].shift()) / dataframe["%-dwt"].shift()
        )
        dataframe["%-dwt_profit"] = dataframe["%-dwt_gain"].clip(lower=0.0)
        dataframe["%-dwt_loss"] = dataframe["%-dwt_gain"].clip(upper=0.0)

        dataframe["%-dwt_profit_mean"] = dataframe["%-dwt_profit"].rolling(self.win_size).mean()
        dataframe["%-dwt_profit_std"] = dataframe["%-dwt_profit"].rolling(self.win_size).std()
        dataframe["%-dwt_loss_mean"] = dataframe["%-dwt_loss"].rolling(self.win_size).mean()
        dataframe["%-dwt_loss_std"] = dataframe["%-dwt_loss"].rolling(self.win_size).std()

        # (Local) Profit & Loss thresholds are used extensively, do not remove!
        dataframe["%%-profit_threshold"] = dataframe[
            "%-dwt_profit_mean"
        ] + self.ts_n_profit_std.value * abs(dataframe["%-dwt_profit_std"])

        dataframe["%%-loss_threshold"] = dataframe[
            "%-dwt_loss_mean"
        ] - self.ts_n_loss_std.value * abs(dataframe["%-dwt_loss_std"])

        # Sequences of consecutive up/downs
        dataframe["%-dwt_dir"] = 0.0
        dataframe["%-dwt_dir"] = np.where(dataframe["%-dwt"].diff() > 0, 1.0, -1.0)

        dataframe["%-dwt_dir_up"] = dataframe["%-dwt_dir"].clip(lower=0.0)
        dataframe["%-dwt_nseq_up"] = dataframe["%-dwt_dir_up"] * (
            dataframe["%-dwt_dir_up"]
            .groupby((dataframe["%-dwt_dir_up"] != dataframe["%-dwt_dir_up"].shift()).cumsum())
            .cumcount()
            + 1
        )
        dataframe["%-dwt_nseq_up"] = dataframe["%-dwt_nseq_up"].clip(
            lower=0.0, upper=20.0
        )  # removes startup artifacts

        dataframe["%-dwt_dir_dn"] = abs(dataframe["%-dwt_dir"].clip(upper=0.0))
        dataframe["%-dwt_nseq_dn"] = dataframe["%-dwt_dir_dn"] * (
            dataframe["%-dwt_dir_dn"]
            .groupby((dataframe["%-dwt_dir_dn"] != dataframe["%-dwt_dir_dn"].shift()).cumsum())
            .cumcount()
            + 1
        )
        dataframe["%-dwt_nseq_dn"] = dataframe["%-dwt_nseq_dn"].clip(lower=0.0, upper=20.0)

        # rolling linear slope of the DWT (i.e. average trend) of near-past
        dataframe["%-dwt_slope"] = dataframe["%-dwt"].rolling(window=6).apply(self.roll_get_slope)

        # moving averages
        dataframe["%-sma"] = ta.SMA(dataframe, timeperiod=self.win_size)
        dataframe["%-ema"] = ta.EMA(dataframe, timeperiod=self.win_size)
        dataframe["%-tema"] = ta.TEMA(dataframe, timeperiod=self.win_size)

        # Bollinger Bands (must include these)
        bollinger = qtpylib.bollinger_bands(dataframe["close"], window=20, stds=2)
        dataframe["%%-bb_lowerband"] = bollinger["lower"]
        dataframe["%%-bb_middleband"] = bollinger["mid"]
        dataframe["%%-bb_upperband"] = bollinger["upper"]
        dataframe["%%-bb_width"] = (
            dataframe["%%-bb_upperband"] - dataframe["%%-bb_lowerband"]
        ) / dataframe["%%-bb_middleband"]
        dataframe["%%-bb_gain"] = (dataframe["%%-bb_upperband"] - dataframe["close"]) / dataframe[
            "close"
        ]
        dataframe["%%-bb_loss"] = (dataframe["%%-bb_lowerband"] - dataframe["close"]) / dataframe[
            "close"
        ]

        # Donchian Channels
        dataframe["%-dc_upper"] = ta.MAX(dataframe["high"], timeperiod=self.win_size)
        dataframe["%-dc_lower"] = ta.MIN(dataframe["low"], timeperiod=self.win_size)
        dataframe["%-dc_mid"] = ta.TEMA(
            ((dataframe["%-dc_upper"] + dataframe["%-dc_lower"]) / 2), timeperiod=self.win_size
        )

        dataframe["%-dcbb_dist_upper"] = dataframe["%-dc_upper"] - dataframe["%%-bb_upperband"]
        dataframe["%-dcbb_dist_lower"] = dataframe["%-dc_lower"] - dataframe["%%-bb_lowerband"]

        # Fibonacci Levels (of Donchian Channel)
        dataframe["%-dc_dist"] = dataframe["%-dc_upper"] - dataframe["%-dc_lower"]

        # Keltner Channels (these can sometimes produce inf results)
        keltner = qtpylib.keltner_channel(dataframe)
        dataframe["%-kc_upper"] = keltner["upper"]
        dataframe["%-kc_lower"] = keltner["lower"]
        dataframe["%-kc_mid"] = keltner["mid"]

        # Stochastic
        period = 14
        smoothD = 3
        SmoothK = 3
        stochrsi = (dataframe["%-rsi"] - dataframe["%-rsi"].rolling(period).min()) / (
            dataframe["%-rsi"].rolling(period).max() - dataframe["%-rsi"].rolling(period).min()
        )
        dataframe["%-srsi_k"] = stochrsi.rolling(SmoothK).mean() * 100
        dataframe["%-srsi_d"] = dataframe["%-srsi_k"].rolling(smoothD).mean()

        # SMA
        dataframe["%-sma_200"] = ta.SMA(dataframe, timeperiod=200)

        # ADX
        dataframe["%-adx"] = ta.ADX(dataframe)

        # Plus Directional Indicator / Movement
        dataframe["%-dm_plus"] = ta.PLUS_DM(dataframe)
        dataframe["%-di_plus"] = ta.PLUS_DI(dataframe)

        # Minus Directional Indicator / Movement
        dataframe["%-dm_minus"] = ta.MINUS_DM(dataframe)
        dataframe["%-di_minus"] = ta.MINUS_DI(dataframe)
        dataframe["%-dm_delta"] = dataframe["%-dm_plus"] - dataframe["%-dm_minus"]
        dataframe["%-di_delta"] = dataframe["%-di_plus"] - dataframe["%-di_minus"]

        # Volume Flow Indicator (MFI) for volume based on the direction of price movement
        dataframe["%-vfi"] = fta.VFI(dataframe, period=14)

        # ATR
        dataframe["%-atr"] = ta.ATR(dataframe, timeperiod=self.win_size)

        # Hilbert Transform Indicator - SineWave
        hilbert = ta.HT_SINE(dataframe)
        dataframe["%-htsine"] = hilbert["sine"]
        dataframe["%-htleadsine"] = hilbert["leadsine"]

        # EWO
        dataframe["%-ewo"] = self.ewo(dataframe, 50, 200)

        # Ultimate Oscillator
        dataframe["%-uo"] = ta.ULTOSC(dataframe)

        # Aroon, Aroon Oscillator
        aroon = ta.AROON(dataframe)
        dataframe["%-aroonup"] = aroon["aroonup"]
        dataframe["%-aroondown"] = aroon["aroondown"]
        dataframe["%-aroonosc"] = ta.AROONOSC(dataframe)

        # Awesome Oscillator
        dataframe["%-ao"] = qtpylib.awesome_oscillator(dataframe)

        # Commodity Channel Index: values [Oversold:-100, Overbought:100]
        dataframe["%-cci"] = ta.CCI(dataframe)

        # Legendary TA indicators
        leg_df = dataframe.copy()
        leg_df = fisher_cg(leg_df)
        leg_df = exhaustion_bars(leg_df)
        leg_df = smi_momentum(leg_df)
        leg_df = pinbar(leg_df, leg_df["smi"])
        leg_df = breakouts(leg_df)

        # Alle Legendary TA Spalten auf einmal hinzufügen um Fragmentierung zu vermeiden
        legendary_columns = {
            "%-smi": leg_df["smi"],
            "%-fisher_cg": leg_df["fisher_cg"],
            "%-fisher_sig": leg_df["fisher_sig"],
            "%-leledc_major": leg_df["leledc_major"],
            "%-leledc_minor": leg_df["leledc_minor"],
            "%-pinbar_buy": leg_df["pinbar_buy"],
            "%-pinbar_sell": leg_df["pinbar_sell"],
            "%-support_level": leg_df["support_level"],
            "%-resistance_level": leg_df["resistance_level"],
            "%-support_breakout": leg_df["support_breakout"],
            "%-resistance_breakout": leg_df["resistance_breakout"],
            "%-support_retest": leg_df["support_retest"],
            "%-potential_support_retest": leg_df["potential_support_retest"],
        }

        # Alle Spalten auf einmal mit pd.concat hinzufügen
        legendary_df = pd.DataFrame(legendary_columns, index=dataframe.index)
        dataframe = pd.concat([dataframe, legendary_df], axis=1)

        # Microstructure Daten hinzufügen
        if self.use_micro.value:
            dataframe = self._merge_micro(dataframe, metadata["pair"])

            # Normalisierung an ATR/Volumen
            if "%-atr" in dataframe.columns:
                dataframe["%-trades_rel_atr"] = dataframe["%-trades_sum"] / (
                    dataframe["%-atr"].replace(0, 1e-9)
                )
                dataframe["%-oi_rel_atr"] = dataframe["%-oi"] / (
                    dataframe["%-atr"].rolling(20).mean().replace(0, 1e-9)
                )
            if "volume" in dataframe.columns:
                basevol = dataframe["volume"].rolling(20).mean().replace(0, 1e-9)
                dataframe["%-trades_rel_vol"] = dataframe["%-trades_sum"] / basevol

        # Liquidation Daten hinzufügen
        if self.use_liq.value:
            dataframe = self._merge_liq(dataframe, metadata["pair"])
            if "%-atr" in dataframe.columns:
                dataframe["%-liq_rel_atr"] = dataframe["%-liq_notional"] / (
                    dataframe["%-atr"].replace(0, 1e-9)
                )
            if "volume" in dataframe.columns:
                dataframe["%-liq_rel_vol"] = dataframe["%-liq_notional"] / (
                    dataframe["volume"].rolling(20).mean().replace(0, 1e-9)
                )

        atr = dataframe["%-atr"] if "%-atr" in dataframe else dataframe["atr"]
        if "close" in dataframe:
            floor = (dataframe["close"] * 1e-6).clip(lower=1e-9)
        else:
            floor = pd.Series(1e-6, index=dataframe.index)
        atr_safe = pd.concat([atr.abs(), floor], axis=1).max(axis=1)

        for base, relcol in [
            ("%-trades_sum", "%-trades_rel_atr"),
            ("%-oi", "%-oi_rel_atr"),
            ("%-liq_notional", "%-liq_rel_atr"),
        ]:
            if base in dataframe and relcol in dataframe:
                dataframe[relcol] = safe_div(
                    dataframe[base].astype("float64"), atr_safe.astype("float64")
                )

        heavy_cols = [
            c
            for c in ["%-trades_sum", "%-oi", "%-oi_delta", "%-oi_ema", "%-liq_notional"]
            if c in dataframe
        ]
        for c in heavy_cols:
            dataframe[c] = robust_log1p_signed(pd.to_numeric(dataframe[c], errors="coerce"))

        clip_quantile(dataframe, heavy_cols, q_hi=0.999, q_lo=0.001)

        rel_cols = [
            c for c in ["%-trades_rel_atr", "%-oi_rel_atr", "%-liq_rel_atr"] if c in dataframe
        ]
        for c in rel_cols:
            dataframe[c] = robust_z(pd.to_numeric(dataframe[c], errors="coerce"), win=200)

        for c in rel_cols:
            if c in dataframe:
                dataframe[c] = dataframe[c].clip(-8.0, 8.0)

        dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)
        dataframe.fillna(0.0, inplace=True)

        return dataframe

    def populate_indicators(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # diag_nonfinite(df)

        drop_cols = ["open_count", "high_count", "low_count", "close_count"]
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")

        df = self.freqai.start(df, metadata, self)

        # NaN/Inf Werte bereinigen
        num = df.select_dtypes(include=[np.number]).columns
        df[num] = df[num].replace([np.inf, -np.inf], 0.0)
        df[num] = df[num].fillna(0.0)

        return df

    def set_freqai_targets(self, df: pd.DataFrame, metadata: dict, **kwargs) -> pd.DataFrame:
        """
        FreqAI Target-Spalten definieren
        Dies ist der FreqAI-Ersatz für die TrainingSignals
        """

        # Lookahead für Target-Berechnung
        lookahead = self.freqai_info["feature_parameters"]["label_period_candles"]

        # Zukünftige Gewinne berechnen (entspricht gain in der ursprünglichen Strategie)
        df["future_gain"] = 100.0 * (df["close"].shift(-lookahead) - df["close"]) / df["close"]

        # Target-Spalte für FreqAI (Regression)
        df["&s-gain"] = df["future_gain"]

        # NaN Werte bereinigen
        df["&s-gain"] = df["&s-gain"].fillna(0.0)

        return df

    def populate_entry_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # ATR (True Range, EMA-glättung)
        tr = pd.concat(
            [
                (df["high"] - df["low"]).abs(),
                (df["high"] - df["close"].shift()).abs(),
                (df["low"] - df["close"].shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["atr"] = tr.ewm(alpha=1 / (int(self.ce_len.value)), adjust=False).mean()

        # HH/LL ohne Lookahead
        df["hh"] = (
            df["high"]
            .rolling(int(self.ce_len.value), min_periods=int(self.ce_len.value))
            .max()
            .shift(1)
        )
        df["ll"] = (
            df["low"]
            .rolling(int(self.ce_len.value), min_periods=int(self.ce_len.value))
            .min()
            .shift(1)
        )

        # Bänder
        df["ce_long"] = df["hh"] - df["atr"] * float(self.ce_mult.value)
        df["ce_short"] = df["ll"] + df["atr"] * float(self.ce_mult.value)

        # atr_sl
        df["atr_short_sl"] = df["close"] + (df["atr"] * float(self.ce_mult.value))
        df["atr_long_sl"] = df["close"] - (df["atr"] * float(self.ce_mult.value))

        # atr_tp1
        df["atr_short_tp1"] = df["close"] - (
            df["atr"] * float(self.ce_mult.value) * float(self.rr_target.value)
        )
        df["atr_long_tp1"] = df["close"] + (
            df["atr"] * float(self.ce_mult.value) * float(self.rr_target.value)
        )

        """Entry-Signale basierend auf FreqAI Vorhersagen mit Guard-Conditions"""

        # Initialisiere Spalten
        df.loc[:, "enter_tag"] = ""
        df["enter_long"] = 0
        df["buy_region"] = 0
        df["curr_target"] = 0.0

        # FreqAI Vorhersage-Spalte (wird automatisch von FreqAI erstellt)
        prediction_col = "&s-gain"

        # Target-Berechnung auf Basis realisierter Gewinne (analog NNPredict)
        profit_nstd = float(self.ts_n_profit_std.value)
        loss_nstd = float(self.ts_n_loss_std.value)
        win_size = max(self.freqai_info["feature_parameters"]["label_period_candles"], 6)

        # Realisierte Gewinne aus Preisreihe (Trailing über Label-Periode)
        lookback = int(self.freqai_info["feature_parameters"]["label_period_candles"])
        realized_gain = (
            100.0 * (df["close"] - df["close"].shift(lookback)) / df["close"].shift(lookback)
        )
        realized_gain = realized_gain.fillna(0.0)
        realized_profit = realized_gain.clip(lower=0.0)
        realized_loss = realized_gain.clip(upper=0.0)

        df["target_profit"] = realized_profit.rolling(
            win_size, min_periods=win_size
        ).mean() + profit_nstd * realized_profit.rolling(win_size, min_periods=win_size).std(ddof=0)

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
            print(
                f"    *** {pair} Trade abgelehnt. Rate ({rate:.2f}) über Ziel ({curr_target:.2f}) "
            )
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

    def ewo(self, dataframe, sma1_length=5, sma2_length=35):
        sma1 = ta.EMA(dataframe, timeperiod=sma1_length)
        sma2 = ta.EMA(dataframe, timeperiod=sma2_length)
        smadif = (sma1 - sma2) / dataframe["close"] * 100
        return smadif

    # Williams %R
    def williams_r(self, dataframe: DataFrame, period: int = 14) -> Series:
        """Williams %R, or just %R, is a technical analysis oscillator showing the current closing price in relation to the high and low
        of the past N days (for a given N). It was developed by a publisher and promoter of trading materials, Larry Williams.
        Its purpose is to tell whether a stock or commodity market is trading near the high or the low, or somewhere in between,
        of its recent trading range.
        The oscillator is on a negative scale, from −100 (lowest) up to 0 (highest).
        """

        highest_high = dataframe["high"].rolling(center=False, window=period).max()
        lowest_low = dataframe["low"].rolling(center=False, window=period).min()

        WR = Series(
            (highest_high - dataframe["close"]) / (highest_high - lowest_low),
            name=f"{period} Williams %R",
        )

        return WR * -100

        # smooth a series

    def smooth(self, y, window):
        box = np.ones(window) / window
        y_smooth = np.convolve(y, box, mode="same")
        # Hack: constrain to 3 decimal places (should be elsewhere, but convenient here)
        y_smooth = np.round(y_smooth, decimals=3)
        return np.nan_to_num(y_smooth)

    # returns (rolling) smoothed version of input column
    def roll_smooth(self, col) -> float:
        # must return scalar, so just calculate prediction and take last value

        smooth = self.smooth(col, 4)
        # smooth = gaussian_filter1d(col, 4)
        # smooth = gaussian_filter1d(col, 2)

        length = len(smooth)
        if length > 0:
            return smooth[length - 1]
        else:
            print("model:", smooth)
            return 0.0

    def get_dwt(self, col):
        a = np.array(col)

        # de-trend the data
        w_mean = a.mean()
        w_std = a.std()
        a_notrend = (a - w_mean) / w_std
        # a_notrend = a_notrend.clip(min=-3.0, max=3.0)

        # get DWT model of data
        restored_sig = self.dwtModel(a_notrend)

        # re-trend
        model = (restored_sig * w_std) + w_mean

        return model

    def roll_get_dwt(self, col) -> float:
        # must return scalar, so just calculate prediction and take last value

        model = self.get_dwt(col)

        length = len(model)
        if length > 0:
            return model[length - 1]
        else:
            # cannot calculate DWT (e.g. at startup), just return original value
            return col[len(col) - 1]

    def dwtModel(self, data):
        # the choice of wavelet makes a big difference
        # for an overview, check out: https://www.kaggle.com/theoviel/denoising-with-direct-wavelet-transform
        # wavelet = 'db1'
        wavelet = "db8"
        # wavelet = 'bior1.1'
        # wavelet = 'haar'  # deals well with harsh transitions
        level = 1
        wmode = "smooth"
        tmode = "hard"
        length = len(data)

        # Apply DWT transform
        coeff = pywt.wavedec(data, wavelet, mode=wmode)

        # remove higher harmonics
        std = np.std(coeff[level])
        sigma = (1 / 0.6745) * self.madev(coeff[-level])
        # sigma = madev(coeff[-level])
        uthresh = sigma * np.sqrt(2 * np.log(length))

        coeff[1:] = (pywt.threshold(i, value=uthresh, mode=tmode) for i in coeff[1:])

        # inverse DWT transform
        model = pywt.waverec(coeff, wavelet, mode=wmode)

        # there is a known bug in waverec where odd numbered lengths result in an extra item at the end
        diff = len(model) - len(data)
        return model[0 : len(model) - diff]
        # return model[diff:]

    def madev(self, d, axis=None):
        """Mean absolute deviation of a signal"""
        return np.mean(np.absolute(d - np.mean(d, axis)), axis)

    def roll_get_slope(self, col) -> float:
        # must return scalar, so just calculate prediction and take last value

        slope = np.polyfit(col.index, col, 1)[0]

        if np.isnan(slope) or np.isinf(slope):
            slope = 10.0

        if (slope < 0) and math.isinf(slope):
            slope = -10.0

        return slope

    def order_filled(self, trade: Trade, order: "Order", current_time: datetime, **kwargs) -> None:
        # Nur defensive Checks – keine teuren Berechnungen
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
            if dataframe is None or dataframe.empty:
                return

            # Entry-Fill? -> TP1/SL aus Signal-Candle in Trade speichern
            if getattr(order, "ft_order_side", None) == "entry":
                # Deine Logik nutzt die Candle VOR entry_time:
                entry_time = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
                signal_time = entry_time - timedelta(minutes=int(self.timeframe_minutes))
                row = dataframe.loc[dataframe["date"] == signal_time]
                if row is None or row.empty:
                    return
                c = row.iloc[-1].squeeze()
                short_sl = c["close"] + (float(c["atr"]) * float(self.atr_mult.value))
                long_sl = c["close"] - (float(c["atr"]) * float(self.atr_mult.value))
                short_tp1 = c["close"] - (
                    float(c["atr"]) * float(self.atr_mult.value) * float(self.rr_target.value)
                )
                long_tp1 = c["close"] + (
                    float(c["atr"]) * float(self.atr_mult.value) * float(self.rr_target.value)
                )

                if trade.is_short:
                    tp1 = float(short_tp1)
                    sl_a = float(short_sl)
                else:
                    tp1 = float(long_tp1)
                    sl_a = float(long_sl)

                trade.set_custom_data(key="tp1_price", value=tp1)
                trade.set_custom_data(key="init_sl_abs", value=sl_a)
                trade.set_custom_data(key="tp1_done", value=False)
                trade.set_custom_data(key="chandelier_active", value=False)

            # Exit-Fill? -> TP1-Flag setzen, falls Tag 'tp1'
            if getattr(order, "ft_order_side", None) == "exit":
                tag = getattr(order, "ft_order_tag", "") or ""
                if tag == "tp1":
                    trade.set_custom_data(key="tp1_done", value=True)
                    trade.set_custom_data(key="tp1_placed", value=False)

        except Exception as e:
            self.dp.send_msg(f"order_filled error {trade.pair}: {e}")

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: Optional[float],
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> Optional[tuple[float, str]]:
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return None

        entry_time = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        signal_time = entry_time - timedelta(minutes=int(self.timeframe_minutes))

        signal_candle = dataframe.loc[dataframe["date"] == signal_time].iloc[-1].squeeze()
        short_tp1 = signal_candle["close"] - (
            float(signal_candle["atr"]) * float(self.atr_mult.value) * float(self.rr_target.value)
        )
        long_tp1 = signal_candle["close"] + (
            float(signal_candle["atr"]) * float(self.atr_mult.value) * float(self.rr_target.value)
        )

        if trade.is_short:
            tp1 = float(short_tp1)
        else:
            tp1 = float(long_tp1)

        # Keine Aktion, wenn bereits ein Order offen (Framework regelt das) oder TP1 schon erledigt
        tp1_done = bool(trade.get_custom_data("tp1_done", False))
        tp1_placed = bool(trade.get_custom_data("tp1_placed", False))
        if tp1_done or tp1_placed:
            return None

        # Long: tp1 erreicht/überschritten; Short: tp1 unterschritten/erreicht
        if trade.nr_of_successful_exits >= 1:
            return None

        hit = (current_rate >= tp1 and not trade.is_short) or (
            current_rate <= tp1 and trade.is_short
        )
        if not hit:
            return None

        # 33% der Position als Stake-Wert reduzieren => -0.33 * stake_amount
        # (Freqtrade rechnet das sauber in Base-Amount um)
        trade.set_custom_data(key="tp1_placed", value=True)
        return (-0.33 * float(trade.stake_amount), "tp1")

    def _tf_seconds(self, tf: str) -> int:
        """
        Robust gegen 5m/15m/1h/4h/1d usw.
        Nutzt pandas, fällt bei exotischen Strings auf Regex zurück.
        """
        try:
            return int(pd.Timedelta(tf).total_seconds())
        except Exception:
            m = re.fullmatch(r"(\d+)([smhdw])", tf.strip().lower())
            if not m:
                raise ValueError(f"Unsupported timeframe: {tf}")
            n, u = int(m.group(1)), m.group(2)
            mult = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
            return n * mult[u]

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool = False,
        **kwargs,
    ) -> float | None:
        # === Tunables ===
        ch_n = self.ce_len.value
        ch_k = self.ce_mult.value  # ATR-Multiplikator
        be_plus = getattr(self, "be_plus", 0.001)  # BE+Offset ~0.1%
        min_trail = getattr(self, "trail_min_pct", 0.012)  # Mindest-Trail (1.2%)
        arm_bars = getattr(self, "ch_arm_bars", 2)  # Mind. Bars seit Entry bevor Trail aktiv
        arm_r = getattr(self, "ch_arm_r", 0.25)  # Mind. 0.25R in Gewinnrichtung bevor Trail aktiv

        # === State ===
        tp1_placed = bool(trade.get_custom_data("tp1_placed", False))
        init_sl_abs = trade.get_custom_data("init_sl_abs", None)

        # --- Helper: DF + Slices ---
        df, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if df is None or df.empty:
            return None  # keine Änderung

        entry_ts = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        sub = df[df["date"] >= entry_ts]
        if sub.empty:
            sub = df.tail(ch_n)
        bars_since_entry = len(sub)

        # === 1) Vor TP1: KEIN initialer SL mehr via custom_stoploss (nur Trailing hier) ===
        if trade.nr_of_successful_exits == 0:
            return None

        # check 2: is tp1_done? if not, return None
        if not tp1_placed:
            return None

        # === 2) Nach TP1: Chandelier-Trail mit Arming + Guards ===
        # 2a) Direkt nach TP1-Fill: einmalig auf BE (+Offset) ziehen
        if after_fill and trade.nr_of_successful_exits == 1:
            be_sl = stoploss_from_open(
                be_plus, current_profit, is_short=trade.is_short, leverage=trade.leverage
            )
            self.dp.send_msg(f"TP1 done - setting BE+offset SL: {be_sl:.4f} for {pair}")
            trade.set_custom_data(key="chandelier_active", value=True)
            return be_sl

        # 2b) Arming-Checks: genug Bars und genug günstige Bewegung?
        use_trailing = True

        # Erst nach genug Bars seit Entry
        if bars_since_entry < arm_bars:
            use_trailing = False

        # R-Multiple Check: genug Bewegung in günstige Richtung?
        if use_trailing and init_sl_abs is not None and trade.open_rate:
            if trade.is_short:
                risk_abs = max(1e-12, init_sl_abs - trade.open_rate)
                fav_abs = max(0.0, trade.open_rate - current_rate)
            else:
                risk_abs = max(1e-12, trade.open_rate - init_sl_abs)
                fav_abs = max(0.0, current_rate - trade.open_rate)
            if fav_abs < arm_r * risk_abs:
                use_trailing = False

        # Wenn Arming-Bedingungen nicht erfüllt: aktuellen SL beibehalten
        if not use_trailing:
            # self.dp.send_msg(f"Trailing not armed yet for {pair} - bars:{bars_since_entry}/{arm_bars}")
            return None

        # 2c) Fenster begrenzen
        if len(sub) > ch_n:
            sub = sub.tail(ch_n)

        # 2d) ATR wählen (robuster: Mittel der letzten ch_n in sub)
        atr_series = sub["atr"].dropna()
        atr_val = float(atr_series.tail(min(len(atr_series), ch_n)).mean() or 0.0)

        # 2e) Chandelier-Level berechnen
        if trade.is_short:
            ll = float(sub["low"].min())
            stop_abs = ll + ch_k * atr_val
            # Richtungs-Guard: Stop muss ÜBER dem Markt liegen
            if stop_abs <= current_rate:
                # Ungültiges Trailing-Level -> nicht künstlich verengen
                # Vor TP1 wurde bereits oben ATR-SL zurückgegeben.
                # Nach TP1: bisherigen SL beibehalten
                return None
        else:
            hh = float(sub["high"].max())
            stop_abs = hh - ch_k * atr_val
            # Richtungs-Guard: Stop muss UNTER dem Markt liegen
            if stop_abs >= current_rate:
                # Ungültiges Trailing-Level -> nicht künstlich verengen
                return None

        dist = stoploss_from_absolute(
            stop_abs, current_rate, is_short=trade.is_short, leverage=trade.leverage
        )

        # 2f) Mindest-Floor gegen Mikro-Noise
        if dist < min_trail:
            dist = min_trail

        # self.dp.send_msg(f"Trailing SL activated for {pair}: {dist:.4f} (stop_abs: {stop_abs:.6f})")
        trade.set_custom_data(key="chandelier_active", value=True)
        return dist

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[Union[str, bool]]:
        # print("DEBUG: Called custom_exit")
        entry_time = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        cur_time = timeframe_to_prev_date(self.timeframe, current_time)
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        atr_sl = trade.get_custom_data(key="atr_sl", default=None)

        if atr_sl is None:  # is not
            signal_time = entry_time - timedelta(minutes=int(self.timeframe_minutes))
            signal_candle = dataframe.loc[dataframe["date"] == signal_time]
            if not signal_candle.empty:
                signal_candle = signal_candle.iloc[-1].squeeze()
                if trade.is_short:
                    trade.set_custom_data(key="atr_sl", value=(signal_candle["atr_short_sl"]))
                    # trade.set_custom_data(key='atr_roi', value=(signal_candle['close'] - ((signal_candle["donchian_upper"] - signal_candle["close"]) * self.risk_reward_ratio.value)))
                    # trade.set_custom_data(key='atr_sl', value=(signal_candle['donchian_upper']))
                else:
                    trade.set_custom_data(key="atr_sl", value=(signal_candle["atr_long_sl"]))
                    # trade.set_custom_data(key='atr_roi', value=(signal_candle['close'] + ((signal_candle["close"] - signal_candle["donchian_lower"]) * self.risk_reward_ratio.value)))
                    # trade.set_custom_data(key=dataframe["atr"] = pta.atr(dataframe["high"], dataframe["low"], dataframe["close"], self.atr_period.value)'atr_sl', value=(signal_candle['donchian_lower']))

            atr_sl = trade.get_custom_data(key="atr_sl", default=None)

        if cur_time > entry_time:
            current_candle = dataframe.iloc[-1].squeeze()
            # Use ATR
            if atr_sl:
                if trade.is_short:
                    if current_rate >= atr_sl:  # Corrected for short trades
                        return "atr_sl"
                else:
                    if current_rate <= atr_sl:
                        return "atr_sl"
            else:
                current_profit = trade.calc_profit_ratio(current_candle["close"])
                if current_profit >= (self.roi * self.lev.value):
                    return "emergency roi"
                if current_profit <= -(self.stoploss * self.lev.value):
                    return "emergency sl"
        return None


def RMI(dataframe, *, length=20, mom=5):
    """
    Source: https://github.com/freqtrade/technical/blob/master/technical/indicators/indicators.py#L912
    """
    df = dataframe.copy()

    df["maxup"] = (df["close"] - df["close"].shift(mom)).clip(lower=0)
    df["maxdown"] = (df["close"].shift(mom) - df["close"]).clip(lower=0)

    df.fillna(0, inplace=True)

    df["emaInc"] = ta.EMA(df, price="maxup", timeperiod=length)
    df["emaDec"] = ta.EMA(df, price="maxdown", timeperiod=length)

    df["RMI"] = np.where(df["emaDec"] == 0, 0, 100 - 100 / (1 + df["emaInc"] / df["emaDec"]))

    return df["RMI"]


log = logging.getLogger(__name__)


def diag_nonfinite(
    df: pd.DataFrame, csv_path: str = "/home/olav/freqai_diag_window.csv", row_window: int = 3
):
    # Nur numerische Spalten prüfen
    num = df.select_dtypes(include=[np.number]).copy()
    if num.shape[1] == 0:
        log.warning("[DIAG] Keine numerischen Spalten gefunden.")
        return

    arr = num.to_numpy()
    bad_mask = ~np.isfinite(arr)

    if not bad_mask.any():
        log.info("[DIAG] Keine Non-Finites in numerischen Spalten.")
        return

    # betroffene Zeilen/Spalten
    bad_rows_idx = np.where(bad_mask.any(axis=1))[0]
    bad_cols_idx = np.where(bad_mask.any(axis=0))[0]

    bad_cols = [num.columns[i] for i in bad_cols_idx]
    log.warning(
        f"[DIAG] Non-Finites in {len(bad_cols)} Spalten: {bad_cols[:10]}{' ...' if len(bad_cols) > 10 else ''}"
    )

    # per Spalte zählen
    counts = []
    for j in bad_cols_idx:
        col = num.columns[j]
        col_vals = num.iloc[:, j].to_numpy()
        counts.append(
            {
                "column": col,
                "NaN": int(np.isnan(col_vals).sum()),
                "PosInf": int(np.isposinf(col_vals).sum()),
                "NegInf": int(np.isneginf(col_vals).sum()),
            }
        )
    counts_df = pd.DataFrame(counts).sort_values(["PosInf", "NegInf", "NaN"], ascending=False)
    log.warning(f"[DIAG] Top-Problemspalten:\n{counts_df.head(10).to_string(index=False)}")

    # Erste Fundstelle sicher ermitteln
    i0 = int(bad_rows_idx[0])
    j0 = int(bad_cols_idx[0])
    c0 = num.columns[j0]
    v0 = num.iat[i0, j0]
    log.warning(f"[DIAG] Erste Fundstelle: row={i0}, col={c0!r}, value={v0}")

    # Kontextfenster dumpen (nur wenige Spalten, die Probleme machen)
    r0 = slice(max(0, i0 - row_window), min(len(df), i0 + row_window + 1))
    df.loc[r0, bad_cols].to_csv(csv_path, index=False)
    log.warning(f"[DIAG] Kontextdump gespeichert: {csv_path}")


EPS = 1e-12  # numerischer Boden


def safe_div(a: pd.Series, b: pd.Series, eps=EPS) -> pd.Series:
    return (a / (b.where(b.abs() > eps, np.sign(b) * eps))).replace([np.inf, -np.inf], np.nan)


def robust_log1p_signed(s: pd.Series) -> pd.Series:
    return np.sign(s) * np.log1p(s.abs())


def _mad(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def robust_z(s: pd.Series, win: int = 200, eps: float = 1e-9) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    med = s.rolling(win, min_periods=win).median()
    mad = s.rolling(win, min_periods=win).apply(_mad, raw=True)
    scale = (1.4826 * mad).where(lambda v: v > eps, eps)
    z = (s - med) / scale
    return z.replace([np.inf, -np.inf], np.nan)


def clip_quantile(df: pd.DataFrame, cols, q_hi=0.999, q_lo=0.001):
    for c in cols:
        if c in df:
            lo = df[c].quantile(q_lo)
            hi = df[c].quantile(q_hi)
            df[c] = df[c].clip(lower=lo, upper=hi)


def fisher_cg(df: DataFrame, length=20, min_period=10):
    """
    Fisher Stochastic Center of Gravity

    Original Pinescript by dasanc
    https://tradingview.com/script/5BT3a9mJ-Fisher-Stochastic-Center-of-Gravity/

    :return: DataFrame with fisher_cg and fisher_sig column populated
    """

    df["hl2"] = (df["high"] + df["low"]) / 2

    if length < min_period:
        length = min_period

    num = 0.0
    denom = 0.0
    CG = 0.0
    MaxCG = 0.0
    MinCG = 0.0
    Value1 = 0.0
    Value2 = 0.0
    Value3 = 0.0

    for i in range(length):
        num += (1 + i) * df["hl2"].shift(i)
        denom += df["hl2"].shift(i)

    CG = -num / denom + (length + 1) / 2
    MaxCG = CG.rolling(window=length).max()
    MinCG = CG.rolling(window=length).min()

    Value1 = np.where(MaxCG != MinCG, (CG - MinCG) / (MaxCG - MinCG), 0)
    Value2 = (
        4 * Value1 + 3 * np.roll(Value1, 1) + 2 * np.roll(Value1, 2) + np.roll(Value1, 3)
    ) / 10
    Value3 = 0.5 * np.log((1 + 1.98 * (Value2 - 0.5)) / (1 - 1.98 * (Value2 - 0.5)))

    df["fisher_cg"] = pd.Series(Value3)  # Center of Gravity
    df["fisher_sig"] = pd.Series(Value3).shift(1)  # Signal / Trigger

    return df


def breakouts(df: DataFrame, length=20):
    """
    S/R Breakouts and Retests

    Makes it easy to work with Support and Resistance
    Find Retests, Breakouts and the next levels

    :return: DataFrame with event columns populated
    """

    high = df["high"]
    low = df["low"]
    close = df["close"]

    pl = low.rolling(window=length * 2 + 1).min()
    ph = high.rolling(window=length * 2 + 1).max()

    s_yLoc = low.shift(length + 1).where(
        low.shift(length + 1) > low.shift(length - 1), low.shift(length - 1)
    )
    r_yLoc = high.shift(length + 1).where(
        high.shift(length + 1) > high.shift(length - 1), high.shift(length + 1)
    )

    cu = close < s_yLoc.shift(length)
    co = close > r_yLoc.shift(length)

    s1 = (high >= s_yLoc.shift(length)) & (close <= pl.shift(length))
    s2 = (
        (high >= s_yLoc.shift(length))
        & (close >= pl.shift(length))
        & (close <= s_yLoc.shift(length))
    )
    s3 = (high >= pl.shift(length)) & (high <= s_yLoc.shift(length))
    s4 = (high >= pl.shift(length)) & (high <= s_yLoc.shift(length)) & (close < pl.shift(length))

    r1 = (low <= r_yLoc.shift(length)) & (close >= ph.shift(length))
    r2 = (
        (low <= r_yLoc.shift(length))
        & (close <= ph.shift(length))
        & (close >= r_yLoc.shift(length))
    )
    r3 = (low <= ph.shift(length)) & (low >= r_yLoc.shift(length))
    r4 = (low <= ph.shift(length)) & (low >= r_yLoc.shift(length)) & (close > ph.shift(length))

    # Events
    df["support_level"] = pl.diff().where(pl.diff().notna())
    df["resistance_level"] = ph.diff().where(ph.diff().notna())

    # Use the last S/R levels instead of nan
    df["support_level"] = df["support_level"].combine_first(df["support_level"].shift())
    df["resistance_level"] = df["resistance_level"].combine_first(df["resistance_level"].shift())

    df["support_breakout"] = cu
    df["resistance_breakout"] = co
    df["support_retest"] = s1 | s2 | s3 | s4
    df["potential_support_retest"] = s1 | s2 | s3
    df["resistance_retest"] = r1 | r2 | r3 | r4
    df["potential_resistance_retest"] = r1 | r2 | r3

    return df


def pinbar(df: DataFrame, smi=None):
    """
    Pinbar - Price Action Indicator

    Pinbars are an easy but sure indication
    of incoming price reversal.
    Signal confirmation with SMI.

    Pinescript Source by PeterO - Thx!
    https://tradingview.com/script/aSJnbGnI-PivotPoints-with-Momentum-confirmation-by-PeterO/

    :return: DataFrame with buy / sell signals columns populated
    """

    low = df["low"]
    high = df["high"]
    close = df["close"]

    tr = true_range(df)

    if smi is None:
        df = smi_momentum(df)
        smi = df["smi"]

    df["pinbar_sell"] = (
        (high < high.shift(1))
        & (close < high - (tr * 2 / 3))
        & (smi < smi.shift(1))
        & (smi.shift(1) > 40)
        & (smi.shift(1) < smi.shift(2))
    )

    df["pinbar_buy"] = (
        (low > low.shift(1))
        & (close > low + (tr * 2 / 3))
        & (smi.shift(1) < -40)
        & (smi > smi.shift(1))
        & (smi.shift(1) > smi.shift(2))
    )

    return df


def smi_momentum(df: DataFrame, k_length=9, d_length=3):
    """
    The Stochastic Momentum Index (SMI) Indicator was developed by
    William Blau in 1993 and is considered to be a momentum indicator
    that can help identify trend reversal points

    :return: DataFrame with smi column populated
    """

    ll = df["low"].rolling(window=k_length).min()
    hh = df["high"].rolling(window=k_length).max()

    diff = hh - ll
    rdiff = df["close"] - (hh + ll) / 2

    avgrel = rdiff.ewm(span=d_length).mean().ewm(span=d_length).mean()
    avgdiff = diff.ewm(span=d_length).mean().ewm(span=d_length).mean()

    df["smi"] = np.where(avgdiff != 0, (avgrel / (avgdiff / 2) * 100), 0)

    return df


def exhaustion_bars(dataframe, maj_qual=6, maj_len=12, min_qual=6, min_len=12, core_length=4):
    """
    Leledc Exhaustion Bars - Extended
    Infamous S/R Reversal Indicator

    leledc_major (Trend):
     1 Up
    -1 Down

    leledc_minor:
    1 Sellers exhausted
    0 Neutral / Hold
    -1 Buyers exhausted

    Original (MT4) https://www.abundancetradinggroup.com/leledc-exhaustion-bar-mt4-indicator/

    :return: DataFrame with columns populated
    """

    bindex_maj, sindex_maj, trend_maj = 0, 0, 0
    bindex_min, sindex_min = 0, 0

    for i in range(len(dataframe)):
        close = dataframe["close"][i]

        if i < 1 or i - core_length < 0:
            dataframe.loc[i, "leledc_major"] = np.nan
            dataframe.loc[i, "leledc_minor"] = 0
            continue

        bindex_maj, sindex_maj = np.nan_to_num(bindex_maj), np.nan_to_num(sindex_maj)
        bindex_min, sindex_min = np.nan_to_num(bindex_min), np.nan_to_num(sindex_min)

        if close > dataframe["close"][i - core_length]:
            bindex_maj += 1
            bindex_min += 1
        elif close < dataframe["close"][i - core_length]:
            sindex_maj += 1
            sindex_min += 1

        update_major = False
        if (
            bindex_maj > maj_qual
            and close < dataframe["open"][i]
            and dataframe["high"][i] >= dataframe["high"][i - maj_len : i].max()
        ):
            bindex_maj, trend_maj, update_major = 0, 1, True
        elif (
            sindex_maj > maj_qual
            and close > dataframe["open"][i]
            and dataframe["low"][i] <= dataframe["low"][i - maj_len : i].min()
        ):
            sindex_maj, trend_maj, update_major = 0, -1, True

        dataframe.loc[i, "leledc_major"] = (
            trend_maj if update_major else np.nan if trend_maj == 0 else trend_maj
        )

        if (
            bindex_min > min_qual
            and close < dataframe["open"][i]
            and dataframe["high"][i] >= dataframe["high"][i - min_len : i].max()
        ):
            bindex_min = 0
            dataframe.loc[i, "leledc_minor"] = -1
        elif (
            sindex_min > min_qual
            and close > dataframe["open"][i]
            and dataframe["low"][i] <= dataframe["low"][i - min_len : i].min()
        ):
            sindex_min = 0
            dataframe.loc[i, "leledc_minor"] = 1
        else:
            dataframe.loc[i, "leledc_minor"] = 0

    return dataframe


def dynamic_exhaustion_bars(dataframe, window=500):
    """
    Dynamic Leledc Exhaustion Bars -  By nilux
    The lookback length and exhaustion bars adjust dynamically to the market.

    leledc_major (Trend):
     1 Up
    -1 Down

    leledc_minor:
    1 Sellers exhausted
    0 Neutral / Hold
    -1 Buyers exhausted

    :return: DataFrame with columns populated
    """

    dataframe["close_pct_change"] = dataframe["close"].pct_change()
    dataframe["pct_change_zscore"] = qtpylib.zscore(dataframe, col="close_pct_change")
    dataframe["pct_change_zscore_smoothed"] = (
        dataframe["pct_change_zscore"].rolling(window=3).mean()
    )
    dataframe["pct_change_zscore_smoothed"].fillna(1.0, inplace=True)

    # To Do: Improve outlier detection

    zscore = dataframe["pct_change_zscore_smoothed"].to_numpy()
    zscore_multi = np.maximum(np.minimum(5.0 - zscore * 2, 5.0), 1.5)

    maj_qual, min_qual = calculate_exhaustion_candles(dataframe, window, zscore_multi)

    dataframe["maj_qual"] = maj_qual
    dataframe["min_qual"] = min_qual

    maj_len, min_len = calculate_exhaustion_lengths(dataframe)

    dataframe["maj_len"] = maj_len
    dataframe["min_len"] = min_len

    dataframe = populate_leledc_major_minor(dataframe, maj_qual, min_qual, maj_len, min_len)

    return dataframe


def populate_leledc_major_minor(dataframe, maj_qual, min_qual, maj_len, min_len):
    bindex_maj, sindex_maj, trend_maj = 0, 0, 0
    bindex_min, sindex_min = 0, 0

    dataframe["leledc_major"] = np.nan
    dataframe["leledc_minor"] = 0

    for i in range(1, len(dataframe)):
        close = dataframe["close"][i]
        short_length = i if i < 4 else 4

        if close > dataframe["close"][i - short_length]:
            bindex_maj += 1
            bindex_min += 1
        elif close < dataframe["close"][i - short_length]:
            sindex_maj += 1
            sindex_min += 1

        update_major = False
        if (
            bindex_maj > maj_qual[i]
            and close < dataframe["open"][i]
            and dataframe["high"][i] >= dataframe["high"][i - maj_len : i].max()
        ):
            bindex_maj, trend_maj, update_major = 0, 1, True
        elif (
            sindex_maj > maj_qual[i]
            and close > dataframe["open"][i]
            and dataframe["low"][i] <= dataframe["low"][i - maj_len : i].min()
        ):
            sindex_maj, trend_maj, update_major = 0, -1, True

        dataframe.at[i, "leledc_major"] = (
            trend_maj if update_major else np.nan if trend_maj == 0 else trend_maj
        )
        if (
            bindex_min > min_qual[i]
            and close < dataframe["open"][i]
            and dataframe["high"][i] >= dataframe["high"][i - min_len : i].max()
        ):
            bindex_min = 0
            dataframe.at[i, "leledc_minor"] = -1
        elif (
            sindex_min > min_qual[i]
            and close > dataframe["open"][i]
            and dataframe["low"][i] <= dataframe["low"][i - min_len : i].min()
        ):
            sindex_min = 0
            dataframe.at[i, "leledc_minor"] = 1
        else:
            dataframe.at[i, "leledc_minor"] = 0

    return dataframe


def calculate_exhaustion_candles(dataframe, window, multiplier):
    """
    Calculate the average consecutive length of ups and downs to adjust the exhaustion bands dynamically
    To Do: Apply ML (FreqAI) to make prediction
    """
    consecutive_diff = np.sign(dataframe["close"].diff())
    maj_qual = np.zeros(len(dataframe))
    min_qual = np.zeros(len(dataframe))

    for i in range(len(dataframe)):
        idx_range = (
            consecutive_diff[i - window + 1 : i + 1] if i >= window else consecutive_diff[: i + 1]
        )
        avg_consecutive = consecutive_count(idx_range)
        if isinstance(avg_consecutive, np.ndarray):
            avg_consecutive = avg_consecutive.item()
        maj_qual[i] = (
            int(avg_consecutive * (3 * multiplier[i])) if not np.isnan(avg_consecutive) else 0
        )
        min_qual[i] = (
            int(avg_consecutive * (3 * multiplier[i])) if not np.isnan(avg_consecutive) else 0
        )

    return maj_qual, min_qual


def calculate_exhaustion_lengths(dataframe):
    """
    Calculate the average length of peaks and valleys to adjust the exhaustion bands dynamically
    To Do: Apply ML (FreqAI) to make prediction
    """
    high_indices = argrelextrema(dataframe["high"].to_numpy(), np.greater)
    low_indices = argrelextrema(dataframe["low"].to_numpy(), np.less)

    avg_peak_distance = np.mean(np.diff(high_indices))
    std_peak_distance = np.std(np.diff(high_indices))
    avg_valley_distance = np.mean(np.diff(low_indices))
    std_valley_distance = np.std(np.diff(low_indices))

    maj_len = int(avg_peak_distance + std_peak_distance)
    min_len = int(avg_valley_distance + std_valley_distance)

    return maj_len, min_len


def linear_growth(
    start: float, end: float, start_time: int, end_time: int, trade_time: int
) -> float:
    """
    Simple linear growth function. Grows from start to end after end_time minutes (starts after start_time minutes)
    """
    time = max(0, trade_time - start_time)
    rate = (end - start) / (end_time - start_time)

    return min(end, start + (rate * time))


def linear_decay(
    start: float, end: float, start_time: int, end_time: int, trade_time: int
) -> float:
    """
    Simple linear decay function. Decays from start to end after end_time minutes (starts after start_time minutes)
    """
    time = max(0, trade_time - start_time)
    rate = (start - end) / (end_time - start_time)

    return max(end, start - (rate * time))


def true_range(dataframe):
    prev_close = dataframe["close"].shift()
    tr = pd.concat(
        [
            dataframe["high"] - dataframe["low"],
            abs(dataframe["high"] - prev_close),
            abs(dataframe["low"] - prev_close),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def consecutive_count(consecutive_diff):
    return np.mean(np.abs(np.diff(np.where(consecutive_diff != 0))))


def compare(a, b):
    return a > b


def compare_less(a, b):
    return a < b
