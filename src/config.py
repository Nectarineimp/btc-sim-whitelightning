#!/usr/bin/env python3
"""
btc-sim-whitelightning: Configuration and Horizon Metadata
"""

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "output"

# Model execution anchors
CALIBRATION_END_DATE = pd.Timestamp("2026-10-31")
FORECAST_START_MONTH = "2026-11"
FORECAST_HORIZON_MONTHS = 12

# Standard percentile outputs matching tournament schema
QUANTILES = [0.05, 0.16, 0.50, 0.84, 0.95]
QUANTILE_NAMES = ["p05", "p16", "p50", "p84", "p95"]

# Simulation parameters
N_SIMULATIONS = 2000
BLOCK_SIZE_DAYS = 14  # Moving block length preserving volatility clustering