#!/usr/bin/env python3
"""
btc-sim-whitelightning: Conditioned Moving-Block Bootstrap Simulator
Generates 12-month forward stochastic distributions (2026-11 to 2027-10).
"""

from pathlib import Path
import numpy as np
import pandas as pd

from config import (
    CALIBRATION_END_DATE,
    FORECAST_HORIZON_MONTHS,
    N_SIMULATIONS,
    BLOCK_SIZE_DAYS,
    QUANTILES,
    QUANTILE_NAMES,
    OUTPUT_DIR,
)


def extract_historical_blocks(df: pd.DataFrame, block_size: int = BLOCK_SIZE_DAYS):
    """
    Extracts contiguous multi-day return blocks tagged by their liquidity regime.
    Focuses on modern market eras (post-2016) for realistic institutional liquidity.
    """
    # Restrict calibration to modern regimes (Era 2 through Era 4)
    calib_df = df[df["date"] >= pd.Timestamp("2016-01-01")].copy().reset_index(drop=True)

    blocks = []
    regimes = []
    n = len(calib_df)

    for i in range(0, n - block_size + 1):
        block_returns = calib_df.loc[i : i + block_size - 1, "log_ret"].values
        # Dominant regime in this block
        block_regime = calib_df.loc[i : i + block_size - 1, "regime"].mode()[0]
        blocks.append(block_returns)
        regimes.append(block_regime)

    return np.array(blocks), np.array(regimes)


def run_whitelightning_simulation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulates forward price paths from 2026-11-01 through 2027-10-31
    and tabulates monthly Low/High percentiles.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    blocks, regimes = extract_historical_blocks(df, BLOCK_SIZE_DAYS)

    # Establish simulation anchor: Last closing price on October 31, 2026
    anchor_price = df[df["date"] <= CALIBRATION_END_DATE]["price"].iloc[-1]
    print(f"[*] Simulation Anchor Price ({CALIBRATION_END_DATE.strftime('%Y-%m-%d')}): ${anchor_price:,.2f}")

    # Forecast horizon setup: 12 months starting 2026-11-01
    start_sim = pd.Timestamp("2026-11-01")
    sim_dates = pd.date_range(start_sim, periods=365, freq="D")
    total_days = len(sim_dates)
    n_blocks_needed = int(np.ceil(total_days / BLOCK_SIZE_DAYS))

    # Macro Conditioning Vector for 2026-2027:
    # 2026-Q4 / 2027-Q1: Neutral to Expanding liquidity
    # 2027-Q2 / 2027-Q4: Expanding debt rollover impulse
    # Sample probabilities favor expansion regimes (Regime +1 and 0)
    p_weights = np.where(regimes == 1, 0.50, np.where(regimes == 0, 0.35, 0.15))
    p_weights = p_weights / p_weights.sum()

    paths = np.zeros((N_SIMULATIONS, total_days))

    for s in range(N_SIMULATIONS):
        chosen_indices = np.random.choice(len(blocks), size=n_blocks_needed, p=p_weights)
        sim_returns = np.concatenate(blocks[chosen_indices])[:total_days]
        # Cumulative compounding
        paths[s, :] = anchor_price * np.exp(np.cumsum(sim_returns))

    # Aggregate simulation paths by Calendar Month
    months = pd.date_range(start_sim, periods=FORECAST_HORIZON_MONTHS, freq="MS").strftime("%Y-%m").tolist()
    forecast_records = []

    for m_str in months:
        m_mask = [d.strftime("%Y-%m") == m_str for d in sim_dates]
        m_slice = paths[:, m_mask]  # shape: (N_SIMULATIONS, days_in_month)

        # In each simulation path, determine that month's min (Low) and max (High)
        path_lows = np.min(m_slice, axis=1)
        path_highs = np.max(m_slice, axis=1)

        low_pcts = np.percentile(path_lows, [q * 100 for q in QUANTILES])
        high_pcts = np.percentile(path_highs, [q * 100 for q in QUANTILES])

        row = {"Month": m_str}
        for q_name, val in zip(QUANTILE_NAMES, low_pcts):
            row[f"Low_{q_name}"] = float(val)
        for q_name, val in zip(QUANTILE_NAMES, high_pcts):
            row[f"High_{q_name}"] = float(val)

        forecast_records.append(row)

    forecast_df = pd.DataFrame(forecast_records)
    out_csv = OUTPUT_DIR / "monthly_forecast.csv"
    forecast_df.to_csv(out_csv, index=False)
    print(f"\n[✓] Standard tournament forecast written to: {out_csv}")
    return forecast_df