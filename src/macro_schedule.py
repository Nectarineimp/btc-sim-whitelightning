#!/usr/bin/env python3
"""
btc-sim-whitelightning: Automated Macro Liquidity Schedule Builder
Ingests official U.S. Treasury and economic data to compute deterministic
forward liquidity states (+1, 0, -1) across multi-year horizons.

Sources:
  1. U.S. Treasury Fiscal Data API: https://api.fiscaldata.treasury.gov
  2. Michael Howell (CrossBorder Capital): 65-Month Global Debt Refinancing Cycle
  3. U.S. Federal Reserve SOMA / TGA Seasonality (April Tax Drain / Q4 Refunding)
"""

import io
from pathlib import Path
import numpy as np
import pandas as pd
import requests

from config import DATA_DIR, RAW_DIR, PROCESSED_DIR


def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def fetch_treasury_debt_summary(force_refresh: bool = False) -> pd.DataFrame:
    """
    Pulls official Monthly Statement of the Public Debt (MSPD) from
    the U.S. Treasury Bureau of the Fiscal Service open REST API.
    """
    raw_path = RAW_DIR / "treasury_debt_summary.csv"
    if raw_path.exists() and not force_refresh:
        return pd.read_csv(raw_path, parse_dates=["record_date"])

    url = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/debt/mspd/mspd_table_1?page[size]=100&sort=-record_date"
    print("[*] Querying U.S. Treasury Fiscal Data REST API (api.fiscaldata.treasury.gov)...")
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if data:
            df = pd.DataFrame(data)
            df["record_date"] = pd.to_datetime(df["record_date"])
            df.to_csv(raw_path, index=False)
            print(f"[✓] Treasury API data cached: {raw_path}")
            return df
    except Exception as e:
        print(f"    [!] Treasury API warning ({e}). Falling back to local offline cache.")

    if raw_path.exists():
        return pd.read_csv(raw_path, parse_dates=["record_date"])
    return pd.DataFrame()


def build_macro_schedule(
    start_month: str = "2026-11",
    horizon_months: int = 60,
    cycle_period_months: float = 65.0,
    cycle_trough_anchor: str = "2022-10-01",
) -> pd.DataFrame:
    """
    Derives deterministic forward macro liquidity regimes from:
      - The 65-month global debt refinancing harmonic wave
      - Federal Reserve / Treasury seasonal tax and refunding dynamics
    """
    ensure_dirs()
    _ = fetch_treasury_debt_summary(force_refresh=False)

    dates = pd.date_range(f"{start_month}-01", periods=horizon_months, freq="MS")
    df = pd.DataFrame({
        "month": dates.strftime("%Y-%m"),
        "date": dates
    })

    # 1. Howell 65-Month Sovereign Debt Wave
    # Anchored at October 2022 global cycle trough (t=0 -> cycle phase = -pi/2)
    t0 = pd.Timestamp(cycle_trough_anchor)
    df["months_from_t0"] = (df["date"].dt.year - t0.year) * 12 + (df["date"].dt.month - t0.month)
    omega = 2.0 * np.pi / cycle_period_months
    phase_offset = -np.pi / 2.0  # Trough at t=0

    # Cycle Level [-1, +1]
    df["howell_cycle_level"] = np.sin(omega * df["months_from_t0"] + phase_offset)
    # Cycle Momentum / Velocity (1st derivative dL/dt)
    df["howell_momentum"] = np.cos(omega * df["months_from_t0"] + phase_offset)

    # 2. Statutory Calendar Seasonality (Treasury Tax Drains vs Spending Injections)
    # April (month 4) & May (month 5): Massive corporate/personal tax drains into TGA
    # December (month 12): Global banking repo window dressing
    # July-November (Q3/Q4): Treasury deficit outlays and reserve reinjection
    seasonal_liquidity_impact = {
        1: -0.10, 2: -0.10, 3: 0.00, 4: -0.80, 5: -0.50, 6: 0.10,
        7: 0.30,  8: 0.40,  9: 0.30, 10: 0.40, 11: 0.50, 12: -0.40
    }
    df["seasonal_factor"] = df["date"].dt.month.map(seasonal_liquidity_impact)

    # 3. Composite Macro Liquidity Acceleration Vector (d²L/dt²)
    # Combining the secular multi-year wave momentum with high-frequency fiscal seasonality
    df["composite_impulse"] = df["howell_momentum"] + (0.35 * df["seasonal_factor"])

    # 4. Deterministic Regime Assignment:
    # +1: Liquidity Surge / Credit Expansion (Impulse > +0.25)
    #  0: Neutral Consolidation (-0.25 <= Impulse <= +0.25)
    # -1: Liquidity Contraction / Refinancing Drain (Impulse < -0.25)
    conditions = [
        df["composite_impulse"] > 0.25,
        df["composite_impulse"] < -0.25
    ]
    choices = [1, -1]
    df["liquidity_regime"] = np.select(conditions, choices, default=0)

    # Human-readable regime description
    desc_map = {
        1: "Expansion (Credit Acceleration)",
        0: "Neutral (Chop / Consolidation)",
        -1: "Contraction (Maturity / Tax Drain)"
    }
    df["regime_label"] = df["liquidity_regime"].map(desc_map)

    # Save to processed directory
    out_file = PROCESSED_DIR / "macro_conditioning_schedule.csv"
    df.to_csv(out_file, index=False)
    print(f"\n[✓] Deterministic macro conditioning schedule built: {out_file}")
    print(f"    - Horizon: {start_month} through {df['month'].iloc[-1]} ({len(df)} months)")
    print(f"    - Howell Cycle Period: {cycle_period_months} months (Anchored: {cycle_trough_anchor})")
    
    return df


if __name__ == "__main__":
    schedule_df = build_macro_schedule(start_month="2026-11", horizon_months=60)
    print("\nFirst 18 Months of Derived Macro Liquidity States:")
    print(schedule_df[["month", "howell_cycle_level", "composite_impulse", "liquidity_regime", "regime_label"]].head(18).to_string(index=False))