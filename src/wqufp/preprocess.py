#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
FILE: preprocess.py
PROJECT: wqu-final-project
ORIGINAL AUTHOR: ryancardenas
DATE CREATED: 11/13/25

Consolidates FirstRateData options prices, Shiller's index dividends, and Yahoo
Finance's SPX prices, and then computes options Greeks for each record.
"""

# import glob
# import os
# import time
# import warnings
from pathlib import Path

# import numpy as np
import pandas as pd
from dotenv import load_dotenv

# from scipy.optimize import brentq, root_scalar
# from scipy.stats import norm
# from tqdm import tqdm
#
# from wqufp.options import BSCall, BSPut
# from wqufp.yield_curve import RiskFreeYield

load_dotenv()  # Load our secrets from the .env file

DAYS_PER_YEAR = 365

TRG_OPT_DF_DIR = Path("./src/data/opt_df")


def get_shiller_dividend_yield() -> pd.DataFrame:
    """
    Loads Robert J. Shiller's data from disk and uses it to derive SPX dividend yield.
    Source: https://shillerdata.com/

    Returns
    -------
    shiller: pandas.DataFrame
        Annualized dividend yields, interpolated by Shiller to monthly intervals.
    """
    shiller_fn = Path("./src/data/ie_data.xls")
    shiller = pd.read_excel(shiller_fn, skiprows=7, sheet_name="Data", dtype=str)[
        ["Date", "P", "D"]
    ].iloc[:-1]
    shiller["P"] = shiller["P"].astype(float)
    shiller["D"] = shiller["D"].astype(float)
    idx = shiller[shiller["Date"].str.len() == 6].index
    shiller.loc[idx, "Date"] = shiller.loc[idx, "Date"] + "0"
    shiller["datetime"] = pd.to_datetime(shiller["Date"].str.replace(".", "-"))
    shiller = shiller.set_index("datetime")
    shiller = shiller[shiller.index.month.isin([3, 6, 9, 12])]
    shiller["q"] = shiller["D"] / shiller["P"]
    shiller = shiller[["q"]]
    return shiller


def get_index_data() -> pd.DataFrame:
    """Imports SPX historical DATETIME, OPEN, and CLOSE, then joins with Shiller's
    dividend yield (forward filled from quarterly samples, which is representative of
    information available during live trading).

    Returns
    -------
    spx_df: pandas.DataFrame
        Dataframe containing DATETIME, OPEN, CLOSE, and Q (annualized dividend yield)
        for SPX.
    """
    # Import SPX historical prices
    yf_spx_fn = Path("./src/data/yf_spx.csv")
    spx_df = pd.read_csv(yf_spx_fn).set_index("datetime")[["open", "close"]]
    spx_df.index = pd.to_datetime(spx_df.index)

    # Join with Shiller's dividend yield
    shiller = get_shiller_dividend_yield
    spx_df = spx_df.join(shiller, how="left")
    spx_df.loc[:, "q"] = spx_df["q"].ffill()
    spx_df.dropna(inplace=True)
    return spx_df
