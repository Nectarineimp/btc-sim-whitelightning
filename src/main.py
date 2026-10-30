#!/usr/bin/env python3
"""
btc-sim-whitelightning: Master Execution Pipeline
"""

from pathlib import Path
import pandas as pd
from ingest import build_unified_dataset
from liquidity_engine import compute_liquidity_impulse
from simulator import run_whitelightning_simulation
from config import OUTPUT_DIR


def main():
    print("=" * 70)
    print("   BTC-SIM-WHITELIGHTNING: MACRO LIQUIDITY STOCHASTIC ENGINE")
    print("=" * 70)

    # 1. Ingest clean data through October 31, 2026
    unified_df = build_unified_dataset()

    # 2. Compute Liquidity Velocity & Acceleration (d²L/dt²)
    print("\n[*] Computing 1st and 2nd derivatives of macro liquidity...")
    modeled_df = compute_liquidity_impulse(unified_df, window=60)

    # 3. Execute Conditioned Moving-Block Bootstrap
    print("\n[*] Running 2,000-path Liquidity-Conditioned Block Bootstrap...")
    forecast_df = run_whitelightning_simulation(modeled_df)

    # 4. Display Formatted Tournament Table
    print("\n================== FORWARD 12-MONTH TOURNAMENT FORECAST ==================")
    pd.set_option("display.max_columns", 12)
    pd.set_option("display.width", 1000)
    print(forecast_df.to_string(index=False))
    print("==========================================================================")


if __name__ == "__main__":
    main()