#!/usr/bin/env python3
"""
btc-sim-whitelightning: Conditioned Moving-Block Bootstrap Simulator
Generates 12-month forward stochastic distributions conditioned on
deterministic macro liquidity regimes (2026-11 to 2027-10).
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
    PROCESSED_DIR,
    OUTPUT_DIR,
)


def extract_historical_blocks(df: pd.DataFrame, block_size: int = BLOCK_SIZE_DAYS):
    """
    Extracts contiguous multi-day return blocks tagged by their liquidity regime.
    Focuses on modern market eras (post-2016) for realistic institutional liquidity.
    """
    calib_df = df[df["date"] >= pd.Timestamp("2016-01-01")].copy().reset_index(drop=True)

    blocks = []
    regimes = []
    n = len(calib_df)

    for i in range(0, n - block_size + 1):
        block_returns = calib_df.loc[i : i + block_size - 1, "log_ret"].values
        # Dominant regime in this block (+1, 0, or -1)
        block_regime = calib_df.loc[i : i + block_size - 1, "regime"].mode()[0]
        blocks.append(block_returns)
        regimes.append(block_regime)

    return np.array(blocks), np.array(regimes)


def get_macro_conditioning_schedule() -> dict[str, int]:
    """
    Generates the deterministic macro liquidity schedule via macro_schedule.py.
    """
    from macro_schedule import build_macro_schedule
    schedule_df = build_macro_schedule(
        start_month="2026-11",
        horizon_months=FORECAST_HORIZON_MONTHS
    )
    lookup = dict(zip(schedule_df["month"], schedule_df["liquidity_regime"]))
    return lookup


def compute_regime_weights(regimes: np.ndarray, target_regime: int) -> np.ndarray:
    """
    Computes sampling probability distribution for historical blocks
    conditioned on the forward target macro liquidity regime.
    """
    if target_regime == 1:
        # Expansion: favor momentum / credit acceleration
        weights = np.where(regimes == 1, 0.65, np.where(regimes == 0, 0.25, 0.10))
    elif target_regime == -1:
        # Contraction: favor drawdown / liquidity drain blocks
        weights = np.where(regimes == -1, 0.65, np.where(regimes == 0, 0.25, 0.10))
    else:
        # Neutral: favor consolidation / chop blocks
        weights = np.where(regimes == 0, 0.60, np.where(regimes == 1, 0.20, 0.20))

    return weights / weights.sum()


def run_whitelightning_simulation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulates forward price paths from 2026-11-01 through 2027-10-31
    using dynamic, month-by-month macro liquidity conditioning.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    blocks, regimes = extract_historical_blocks(df, BLOCK_SIZE_DAYS)

    # Establish simulation anchor: Last closing price on October 31, 2026
    anchor_price = df[df["date"] <= CALIBRATION_END_DATE]["price"].iloc[-1]
    print(f"[*] Simulation Anchor Price ({CALIBRATION_END_DATE.strftime('%Y-%m-%d')}): ${anchor_price:,.2f}")

    # Load macro regime schedule lookup
    macro_schedule = get_macro_conditioning_schedule()

    # Precompute probability weights for all three possible regimes
    weight_cache = {
        1: compute_regime_weights(regimes, 1),
        0: compute_regime_weights(regimes, 0),
        -1: compute_regime_weights(regimes, -1),
    }

    start_sim = pd.Timestamp("2026-11-01")
    months = pd.date_range(start_sim, periods=FORECAST_HORIZON_MONTHS, freq="MS").strftime("%Y-%m").tolist()

    # Matrix tracking each simulation's current price across transitions
    current_prices = np.full(N_SIMULATIONS, anchor_price)
    forecast_records = []

    print("[*] Stepping through forward macro regimes month-by-month...")

    for m_str in months:
        target_regime = int(macro_schedule.get(m_str, 0))
        regime_desc = {
            1: "Expansion (+1)",
            0: "Neutral (0)",
            -1: "Contraction (-1)"
        }.get(target_regime, "Neutral (0)")

        print(f"    - Month {m_str}: Target Regime = {regime_desc}")
        p_weights = weight_cache[target_regime]

        days_in_month = pd.Period(m_str, freq="M").days_in_month
        n_blocks_needed = int(np.ceil(days_in_month / BLOCK_SIZE_DAYS))

        month_lows = np.zeros(N_SIMULATIONS)
        month_highs = np.zeros(N_SIMULATIONS)

        for s in range(N_SIMULATIONS):
            chosen_indices = np.random.choice(len(blocks), size=n_blocks_needed, p=p_weights)
            sim_returns = np.concatenate(blocks[chosen_indices])[:days_in_month]

            # Compound daily price within the month starting from the previous month's close
            daily_prices = current_prices[s] * np.exp(np.cumsum(sim_returns))

            month_lows[s] = np.min(daily_prices)
            month_highs[s] = np.max(daily_prices)
            current_prices[s] = daily_prices[-1]  # Carry closing price forward

        # Tabulate quantiles for this month matching the tournament schema
        low_pcts = np.percentile(month_lows, [q * 100 for q in QUANTILES])
        high_pcts = np.percentile(month_highs, [q * 100 for q in QUANTILES])

        row = {"Month": m_str}
        for q_name, val in zip(QUANTILE_NAMES, low_pcts):
            row[f"Low_{q_name}"] = float(val)
        for q_name, val in zip(QUANTILE_NAMES, high_pcts):
            row[f"High_{q_name}"] = float(val)

        forecast_records.append(row)

    forecast_df = pd.DataFrame(forecast_records)
    out_csv = OUTPUT_DIR / "monthly_forecast.csv"
    forecast_df.to_csv(out_csv, index=False)
    print(f"\n[✓] Macro-conditioned tournament forecast written to: {out_csv}")
    return forecast_df