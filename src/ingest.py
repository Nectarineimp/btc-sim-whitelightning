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
    """Pulls public FRED economic series, handling 'observation_date', 'DATE', or 'date'."""
    raw_path = RAW_DIR / f"{series_id}.csv"

    def _normalize_fred_df(df: pd.DataFrame) -> pd.DataFrame:
        # Match 'observation_date', 'DATE', or 'date'
        date_col = next(
            (c for c in df.columns if c.lower() in ("observation_date", "date")),
            df.columns[0],  # Fallback to first column
        )
        df = df.rename(columns={date_col: "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
        return df[["date", series_id]].dropna().sort_values("date").reset_index(drop=True)

    # 1. Read existing cached file if valid
    if raw_path.exists() and not force_refresh:
        try:
            df = pd.read_csv(raw_path)
            if series_id in df.columns:
                df = _normalize_fred_df(df)
                # Overwrite cache if it still had the old 'observation_date' header
                df.to_csv(raw_path, index=False)
                return df
        except Exception as e:
            print(f"    [!] Cache read issue for {series_id} ({e}), re-fetching...")

    # 2. Download from FRED
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()

        if "<html" in resp.text.lower():
            raise ValueError(f"FRED endpoint returned HTML instead of CSV for {series_id}")

        df = pd.read_csv(io.StringIO(resp.text))
        df = _normalize_fred_df(df)
        df.to_csv(raw_path, index=False)
        return df

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
        # Merge on full daily price calendar first so weekly series align properly
        liq_df = pd.DataFrame({"date": price_df["date"]})
        liq_df = (
            liq_df.merge(walcl, on="date", how="left")
            .merge(wtregen, on="date", how="left")
            .merge(rrpontsyd, on="date", how="left")
        )
        liq_df = liq_df.sort_values("date").ffill().bfill()

        # Normalize units to Billions USD:
        # WALCL and WTREGEN are in Millions (/ 1000 -> Billions); RRPONTSYD is already in Billions
        walcl_bn = liq_df["WALCL"] / 1000.0
        tga_bn = np.where(liq_df["WTREGEN"].median() > 10000, liq_df["WTREGEN"] / 1000.0, liq_df["WTREGEN"])
        rrp_bn = liq_df["RRPONTSYD"].fillna(0.0)

        liq_df["net_liquidity"] = walcl_bn - tga_bn - rrp_bn
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