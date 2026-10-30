#!/usr/bin/env python3
"""
btc-sim-whitelightning: Ingestion Engine
Fetches raw price history and central bank liquidity plumbing.
"""

import io
from pathlib import Path
import numpy as np
import pandas as pd
import requests

from config import RAW_DIR, PROCESSED_DIR, CALIBRATION_END_DATE


def ensure_directories():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def fetch_coinmetrics_daily(force_refresh: bool = False) -> pd.DataFrame:
    """Fetches continuous daily Bitcoin prices up through calibration end date."""
    raw_path = RAW_DIR / "btc_coinmetrics_raw.csv"

    if raw_path.exists() and not force_refresh:
        df = pd.read_csv(raw_path)
    else:
        url = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(resp.text)
        df = pd.read_csv(io.StringIO(resp.text))

    df["date"] = pd.to_datetime(df["time"]).dt.normalize()
    df["price"] = pd.to_numeric(df["PriceUSD"], errors="coerce")
    df = df[["date", "price"]].dropna().sort_values("date").reset_index(drop=True)

    # Prepend New Liberty Standard (2009-2010) benchmarks
    nls_records = [
        {"date": pd.Timestamp("2009-10-05"), "price": 0.00099},
        {"date": pd.Timestamp("2009-10-12"), "price": 0.00100},
        {"date": pd.Timestamp("2010-02-20"), "price": 0.01000},
        {"date": pd.Timestamp("2010-07-17"), "price": 0.08000},
    ]
    nls_df = pd.DataFrame(nls_records)
    min_date = df["date"].min()
    combined = pd.concat([nls_df[nls_df["date"] < min_date], df]).drop_duplicates(subset=["date"])
    combined = combined.set_index("date").resample("D").interpolate(method="time").reset_index()

    # Cap strictly at October 31, 2026
    combined = combined[combined["date"] <= CALIBRATION_END_DATE].reset_index(drop=True)
    return combined


def fetch_fred_series(series_id: str, force_refresh: bool = False) -> pd.DataFrame:
    """Pulls public FRED economic series."""
    raw_path = RAW_DIR / f"{series_id}.csv"
    if raw_path.exists() and not force_refresh:
        return pd.read_csv(raw_path, parse_dates=["DATE"])

    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(resp.text)
        df = pd.read_csv(io.StringIO(resp.text))
        df["DATE"] = pd.to_datetime(df["DATE"])
        df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
        return df.dropna().rename(columns={"DATE": "date"})
    except Exception as e:
        print(f"    [!] Note: Online FRED pull for {series_id} unavailable ({e}). Using synthetic baseline.")
        return pd.DataFrame()


def build_unified_dataset():
    """Generates the unified continuous dataset ready for simulation."""
    ensure_directories()
    print("[*] Ingesting continuous daily price series...")
    price_df = fetch_coinmetrics_daily(force_refresh=False)

    print("[*] Ingesting Fed Net Liquidity components...")
    walcl = fetch_fred_series("WALCL")        # Total Fed Assets
    wtregen = fetch_fred_series("WTREGEN")    # Treasury General Account
    rrpontsyd = fetch_fred_series("RRPONTSYD")# Reverse Repo

    if not walcl.empty and not wtregen.empty and not rrpontsyd.empty:
        liq_df = walcl.merge(wtregen, on="date", how="outer").merge(rrpontsyd, on="date", how="outer")
        liq_df = liq_df.sort_values("date").interpolate(method="linear")
        liq_df["net_liquidity"] = liq_df["WALCL"] - liq_df["WTREGEN"] - liq_df["RRPONTSYD"]
    else:
        # Fallback: estimate from historical empirical baseline
        liq_df = pd.DataFrame({"date": price_df["date"]})
        liq_df["net_liquidity"] = 6_000_000.0  # Normalized $6.0T baseline

    unified = price_df.merge(liq_df[["date", "net_liquidity"]], on="date", how="left")
    unified["net_liquidity"] = unified["net_liquidity"].interpolate(method="linear")

    out_path = PROCESSED_DIR / "btc_macro_unified.csv"
    unified.to_csv(out_path, index=False)
    print(f"[✓] Unified dataset saved: {out_path} ({len(unified):,} records up through {CALIBRATION_END_DATE.strftime('%Y-%m-%d')})")
    return unified