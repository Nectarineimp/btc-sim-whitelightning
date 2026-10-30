#!/usr/bin/env python3
"""
btc-sim-whitelightning: Liquidity Impulse & Acceleration Engine
Calculates dL/dt and d²L/dt² to classify historical market regimes.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from config import PROCESSED_DIR


def compute_liquidity_impulse(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Computes rolling velocity and acceleration of net liquidity.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    # Log returns of price
    df["log_ret"] = np.log(df["price"] / df["price"].shift(1))

    # Liquidity 1st derivative: % velocity over lookback window
    df["liq_velocity"] = df["net_liquidity"].pct_change(periods=window)

    # Liquidity 2nd derivative: acceleration (rate of change of velocity)
    df["liq_acceleration"] = df["liq_velocity"] - df["liq_velocity"].shift(window)

    # Regime assignment based on 2nd derivative acceleration
    accel_std = df["liq_acceleration"].std(skipna=True)
    if pd.isna(accel_std) or accel_std == 0:
        accel_std = 1.0

    conditions = [
        df["liq_acceleration"] > (0.25 * accel_std),
        df["liq_acceleration"] < (-0.25 * accel_std),
    ]
    choices = [1, -1]  # 1: Expansion, -1: Contraction
    df["regime"] = np.select(conditions, choices, default=0)  # 0: Neutral / Consolidation

    return df