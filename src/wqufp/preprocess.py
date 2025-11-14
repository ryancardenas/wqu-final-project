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

import glob
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.optimize import brentq, root_scalar
from tqdm import tqdm

from wqufp.options import BSCall, BSPut
from wqufp.yield_curve import RiskFreeYield

load_dotenv()  # Load our secrets from the .env file

DAYS_PER_YEAR = 365
TRG_OPT_DF_DIR = Path("./src/data/opt_df")


def get_shiller_dividend_yield() -> pd.DataFrame:
    """
    Loads Robert J. Shiller's data from disk and uses it to derive SPX dividend yield.
    Source: https://shillerdata.com/

    Returns
    -------
    shiller: pd.DataFrame
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
    spx_df: pd.DataFrame
        Dataframe containing DATETIME, OPEN, CLOSE, and Q (annualized dividend yield)
        for SPX.
    """
    # Import SPX historical prices
    yf_spx_fn = Path("./src/data/yf_spx.csv")
    spx_df = pd.read_csv(yf_spx_fn).set_index("datetime")[["open", "close"]]
    spx_df.index = pd.to_datetime(spx_df.index)

    # Join with Shiller's dividend yield
    shiller = get_shiller_dividend_yield()
    spx_df = spx_df.join(shiller, how="left")
    spx_df.loc[:, "q"] = spx_df["q"].ffill()
    spx_df.dropna(inplace=True)
    return spx_df


def load_concatenate_options_files(
    opt_fn_list: list[str], verbose: int = 1
) -> pd.DataFrame:
    """Iterates through each filepath in opt_fn_list, loading the data and filtering
    out invalid deltas/DTE/MIDs, and then concatenates the results.

    Returns
    -------
    opt_df: pd.DataFrame
        Concatenated and filtered options data.
    """
    if verbose >= 1:
        print("Loading options files into memory...")
    opt_df = pd.DataFrame()
    for i in tqdm(range(len(opt_fn_list))):
        tmp = pd.read_csv(opt_fn_list[i], names=cols)

        # Ignore out-of-range deltas
        upper_delta_threshold = tmp["delta"].abs() < 1.0
        lower_delta_threshold = tmp["delta"].abs() > 0.0

        # Filter out 0 DTE options
        not_zero_dte = tmp["trade date"] != tmp["expiry date"]
        tmp = tmp[
            upper_delta_threshold & lower_delta_threshold & not_zero_dte
        ].reset_index(drop=True)
        opt_df = pd.concat([opt_df, tmp], ignore_index=True)

        # Filter invalid MID prices
        opt_df["mid price"] = (opt_df["bid price"] + opt_df["ask price"]) / 2
        opt_df = opt_df[opt_df["mid price"] > 0].reset_index(drop=True)

        opt_df.reset_index(drop=True)  # Just in case...
    return opt_df


def compute_contract_durations_and_types(
    opt_df: pd.DataFrame,
) -> pd.DataFrame:
    """Computes days-to-expiration ('dte') and years-to-expiration ('T') while
    adjusting for standard (monthlies) and non-standard (weeklies) index contracts.
    Handles the obsolete (pre-2016) convention of listing standard expiries on
    Saturday instead of Friday. Labels all third-Friday expiries as 'standard'.

    Returns
    -------
    opt_df: pd.DataFrame
        Options data with dte, T, and standard labels included, and with 'expiry
        date' corrected for the obsolete Saturday convention.
    """
    print("Casting datestrings as datetimes...")
    opt_df["trade date"] = pd.to_datetime(opt_df["trade date"])
    opt_df["expiry date"] = pd.to_datetime(opt_df["expiry date"])

    print("Adjusting for obsolete Saturday expiry convention...")
    sat_idx = opt_df[opt_df["expiry date"].dt.dayofweek == 5].index
    opt_df.loc[sat_idx, "expiry date"] = opt_df.loc[
        sat_idx, "expiry date"
    ] - pd.Timedelta(days=1)

    print("Computing contract duration...")
    opt_df["dte"] = (opt_df["expiry date"] - opt_df["trade date"]).dt.days
    opt_df["T"] = opt_df["dte"] / DAYS_PER_YEAR

    print("Identifying standard monthly contracts...")
    is_friday = opt_df["expiry date"].dt.weekday == 4
    is_third_week = (opt_df["expiry date"].dt.day >= 15) & (
        opt_df["expiry date"].dt.day <= 21
    )
    opt_df["standard"] = is_friday & is_third_week

    return opt_df


def apply_underlying_price_at_expiry(opt_df: pd.DataFrame) -> pd.DataFrame:
    """Computes the underlying settlement price at expiry. Handles standard/non-standard
    expiry settlements.

    Returns
    -------
    opt_df: pd.DataFrame
        Options data with Sf label included.
    """
    print("Identifying SPX CLOSE at expiry date...")
    opt_df = (
        opt_df.set_index("expiry date")
        .join(spx_df["close"].rename("Sf"), how="inner", lsuffix="", rsuffix="_s")
        .reset_index(drop=False)
    )  # Transform back to integer index for next step...

    # Need to operate on integer index; datetime index takes too long to slice
    print("Applying next-day SPX OPEN for standard (monthly) options...")
    standard_idx = opt_df[opt_df["standard"]].index
    opt_df.loc[standard_idx, "Sf"] = (
        opt_df.set_index("expiry date")
        .join(
            spx_df.shift(-1)["open"].rename("Sf"), how="left", lsuffix="", rsuffix="_s"
        )
        .reset_index(drop=False)
        .loc[standard_idx, "Sf"]
    )

    return opt_df


def apply_underlying_price_and_yield_at_entry(opt_df: pd.DataFrame) -> pd.DataFrame:
    """Applies the underlying price and dividend yield at 'trade date'.
    Note: For some reason, after reindexing on 'trade date' for the purposes of
    joining with spx_df, calling reset_index() causes the column to be renamed
    'index'. This does not happen with 'expiry date'. For this reason, we rename the
    'index' column back to 'trade date' after resetting the index.

    Returns
    -------
    opt_df: pd.DataFrame
        Options data with S0 (initial SPX price) and q (annualized dividend yield)
        labels included.
    """
    print("Identifying SPX CLOSE at trade date...")
    opt_df = opt_df.set_index("trade date").join(
        spx_df["close"].rename("S0"), how="left", lsuffix="", rsuffix="_s"
    )
    # Not sure why 'trade date' is special, but it requires renaming
    # from 'index' after joining with spx_df.
    print("Aligning S&P 500 dividend yield...")
    opt_df = (
        opt_df.join(spx_df["q"], how="left", lsuffix="", rsuffix="_s")
        .reset_index(drop=False)
        .rename(columns={"index": "trade date"})
    )
    return opt_df


def apply_risk_free_rate(opt_df: pd.DataFrame, rf_curve: RiskFreeYield) -> pd.DataFrame:
    """Uses wqufp.yield_curve.RiskFreeYield to interpolate the risk-free rate at
    'trade date' based on treasury data collected from the FRED API. Since FRED lists
    interest rates as percentages (1.5%, 2.2%, ...), we divide by 100 to convert to
    decimals (0.015, 0.022, ...).

    Returns
    -------
    opt_df: pd.DataFrame
        Options data with rf (risk-free rate) included.
    """
    print("Computing risk-free rates...")
    for trade_date in opt_df["trade date"].unique():
        idx = opt_df[opt_df["trade date"] == trade_date].index
        opt_df.loc[idx, "rf"] = (
            rf_curve(date=trade_date, ytm=opt_df.loc[idx, "T"]) / 100.0
        )
    return opt_df


def process_call_options(opt_df: pd.DataFrame) -> pd.DataFrame:
    """Applies call-specific logic to compute the following labels:
        M0 = moneyness trade date
        Mf = moneyness at expiry date
        ITM = True if options expires in-the-money; False otherwise
        payoff = contract realized payoff at maturity, in units of [USD]
        R_payoff = return on initial contract price, as a decimal

    Returns
    -------
    call_df: pd.DataFrame
        Call options data, sorted by expiry date and trade date.
    """
    print("Processing Call ITM conditions...")
    call_df = opt_df[opt_df["call/put"] == "c"].reset_index(drop=True)
    call_df["M0"] = call_df["S0"] / call_df["strike"] - 1
    call_df["Mf"] = np.maximum(0.0, (call_df["Sf"] / call_df["strike"]) - 1)
    call_df["ITM"] = call_df["Sf"] > call_df["strike"]
    call_df["payoff"] = np.maximum(0, call_df["Sf"] - call_df["strike"])
    call_df["R_payoff"] = (call_df["payoff"] / call_df["mid price"]) - 1
    call_df.sort_values(
        by=["expiry date", "trade date"], ignore_index=True, inplace=True
    )
    return call_df


def process_put_options(opt_df: pd.DataFrame) -> pd.DataFrame:
    """Applies put-specific logic to compute the following labels:
        M0 = moneyness trade date
        Mf = moneyness at expiry date
        ITM = True if options expires in-the-money; False otherwise
        payoff = contract realized payoff at maturity, in units of [USD]
        R_payoff = return on initial contract price, as a decimal

    Returns
    -------
    put_df: pd.DataFrame
        Put options data, sorted by expiry date and trade date.
    """
    print("Processing Put ITM conditions...")
    put_df = opt_df[opt_df["call/put"] == "p"].reset_index(drop=True)
    put_df["M0"] = 1 - put_df["S0"] / put_df["strike"]
    put_df["Mf"] = np.maximum(0.0, 1 - (put_df["Sf"] / put_df["strike"]))
    put_df["ITM"] = put_df["Sf"] < put_df["strike"]
    put_df["payoff"] = np.maximum(0, put_df["strike"] - put_df["Sf"])
    put_df["R_payoff"] = (put_df["payoff"] / put_df["mid price"]) - 1
    put_df.sort_values(
        by=["expiry date", "trade date"], ignore_index=True, inplace=True
    )
    return put_df


def compute_options_greeks(
    opt_df: pd.DataFrame,
    min_vol: float = 0.001,
    max_vol: float = 2.5,
    xtol: float = 1e-6,
    max_iter: int = 100,
) -> pd.DataFrame:
    """Computes sigma (implied volatility), delta, theta, gamma, rho, and vega

    Returns
    -------
    opt_df: pd.DataFrame
        Options data with Greeks included.
    """
    print("Computing options Greeks...")
    for i in tqdm(range(opt_df.shape[0])):
        row = opt_df.iloc[i]

        if row["call/put"] == "c":
            greek = BSCall()
        elif row["call/put"] == "p":
            greek = BSPut()
        else:
            raise ValueError("call/put should be either 'c' or 'p'")

        def f(sigma_: float) -> float:
            y_pred: float = greek.price(
                S0=row["S0"],
                K=row["strike"],
                T=row["T"],
                r=row["rf"],
                sigma=sigma_,
                q=row["q"],
            )
            y_true: float = row["mid price"]
            return y_pred - y_true

        try:
            sigma = brentq(f, min_vol, max_vol, xtol=xtol)
        except (ValueError, RuntimeError):
            try:
                # Try maximum of FRD bid/ask volatilities as initial guess
                x0 = np.maximum(
                    row["ask implied volatility"], row["bid implied volatility"]
                )
                result = root_scalar(
                    f, x0=x0, method="newton", xtol=xtol, maxiter=max_iter
                )
                if result.converged:
                    sigma = result.root
                else:
                    sigma = np.nan
            except (ValueError, RuntimeError):
                sigma = np.nan

        delta, theta, rho, gamma, vega, p_itm = greek.greeks(
            S0=row["S0"],
            K=row["strike"],
            T=row["T"],
            r=row["rf"],
            sigma=sigma,
            q=row["q"],
        )

        opt_df.loc[i, "sigma_"] = sigma
        opt_df.loc[i, "delta_"] = delta
        opt_df.loc[i, "theta_"] = theta
        opt_df.loc[i, "rho_"] = rho
        opt_df.loc[i, "gamma_"] = gamma
        opt_df.loc[i, "vega_"] = vega
        opt_df.loc[i, "pitm_"] = p_itm

    return opt_df


if __name__ == "__main__":
    start = time.perf_counter()

    trg_path_hdf5 = TRG_OPT_DF_DIR / "options.hdf5"

    warnings.filterwarnings("ignore", message="Derivative was zero")
    warnings.filterwarnings(
        "ignore", message="divide by zero encountered in scalar divide"
    )

    spx_df = get_index_data()
    rf_curve = RiskFreeYield()

    cols_str = (
        "Trade Date,Strike,Expiry Date,Call/Put,Last Trade Price,Bid Price,"
        "Ask Price,Bid Implied Volatility,Ask Implied Volatility,"
        "Open Interest,Volume,Delta,Gamma,Vega,Theta,Rho"
    )
    cols = [c.lower().replace(",", "") for c in cols_str.split(",")]

    options_dir = os.getenv("OPTIONS_DIR")
    if options_dir is None:
        raise ValueError("OPTIONS_DIR environment variable must be set")
    all_opt_fn_list = sorted(glob.glob(options_dir + "/SPX/*"))
    data_years = pd.Series([c.split("_")[1] for c in all_opt_fn_list]).unique()

    for data_year in data_years:
        print(f"Processing Year: {data_year}")
        opt_fn_list = sorted(glob.glob(options_dir + f"/SPX/*_{data_year}_*"))

        opt_df = load_concatenate_options_files(opt_fn_list=opt_fn_list)
        opt_df = compute_contract_durations_and_types(opt_df=opt_df)
        opt_df = apply_underlying_price_at_expiry(opt_df=opt_df)
        opt_df = apply_underlying_price_and_yield_at_entry(opt_df=opt_df)
        opt_df = apply_risk_free_rate(opt_df=opt_df, rf_curve=rf_curve)

        call_df = process_call_options(opt_df=opt_df)
        put_df = process_put_options(opt_df=opt_df)

        opt_df = pd.concat([call_df, put_df], ignore_index=True)
        opt_df = compute_options_greeks(opt_df=opt_df)

        print("Saving to disk...")
        trg_path_csv = TRG_OPT_DF_DIR / (data_year + ".csv")
        opt_df.to_csv(trg_path_csv, index=False)
        opt_df.to_hdf(
            path_or_buf=trg_path_hdf5,
            key="options",
            mode="a",
            complevel=9,
            complib="zlib",
            format="table",
            append=True,
        )

        stop = time.perf_counter()
        print(f"Processing time elapsed: {(stop - start) / 60:.2f} minutes")
        print("*" * 50)

    print("*" * 50)
    print("Process Complete!")
