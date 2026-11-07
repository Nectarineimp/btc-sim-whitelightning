#!/usr/bin/env python3
"""
btc-sim-whitelightning: Liquidity Impulse & Acceleration Engine
Calculates lagged dL/dt and d²L/dt² (75-day transmission lag) and
demeans Era drift excess to classify historical market regimes.
"""

import numpy as np
import pandas as pd


def compute_liquidity_impulse(
    df: pd.DataFrame,
    window: int = 60,
    transmission_lag_days: int = 75,
) -> pd.DataFrame:
    """
    Computes rolling velocity and acceleration of net liquidity, shifted forward
    by the empirical ~75-day (10-11 week) macro-to-BTC transmission lag.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    # Raw daily log return
    df["log_ret"] = np.log(df["price"] / df["price"].shift(1)).fillna(0.0)

    # Remove early-era hyper-monetization bias by normalizing post-2016 drift to Era 4 scale
    # Era 4 (2022-2026) baseline drift is ~0.0005/day (~20% annualized baseline)
    post_2016_mask = df["date"] >= pd.Timestamp("2016-01-01")
    era2_3_mask = (df["date"] >= pd.Timestamp("2016-01-01")) & (df["date"] < pd.Timestamp("2022-01-01"))
    if era2_3_mask.any():
        excess_early_drift = df.loc[era2_3_mask, "log_ret"].mean() - 0.00045
        df.loc[era2_3_mask, "log_ret"] = df.loc[era2_3_mask, "log_ret"] - excess_early_drift

    # 1st derivative: % velocity over lookback window
    raw_velocity = df["net_liquidity"].pct_change(periods=window)

    # 2nd derivative: acceleration (change in velocity)
    raw_accel = raw_velocity - raw_velocity.shift(window)

    # Apply the ~75-day macro transmission lag (liquidity today drives BTC in ~75 days)
    df["liq_velocity"] = raw_velocity.shift(transmission_lag_days)
    df["liq_acceleration"] = raw_accel.shift(transmission_lag_days)

    # Composite transmitted macro score (blending lagged velocity and acceleration)
    vel_z = (df["liq_velocity"] - df["liq_velocity"].mean()) / (df["liq_velocity"].std() + 1e-8)
    acc_z = (df["liq_acceleration"] - df["liq_acceleration"].mean()) / (df["liq_acceleration"].std() + 1e-8)
    ret_14d = df["log_ret"].rolling(14).sum()
    ret_z = (ret_14d - ret_14d.mean()) / (ret_14d.std() + 1e-8)

    # Coupled macro-transmission regime score
    regime_score = 0.45 * vel_z.fillna(0) + 0.35 * acc_z.fillna(0) + 0.20 * ret_z.fillna(0)

    conditions = [
        regime_score > 0.25,
        regime_score < -0.25,
    ]
    choices = [1, -1]  # +1: Expansion, -1: Contraction
    df["regime"] = np.select(conditions, choices, default=0)

    # Diagnostic summary of regime block returns
    calib = df[post_2016_mask]
    for r, label in [(1, "Expansion (+1)"), (0, "Neutral (0)"), (-1, "Contraction (-1)")]:
        sub = calib[calib["regime"] == r]["log_ret"]
        ann_ret = (np.exp(sub.mean() * 365) - 1) * 100
        print(f"    [Regime {r:+d}] {label:<18} | Days: {len(sub):4d} | Mean Annualized Drift: {ann_ret:+6.1f}%")

    return df