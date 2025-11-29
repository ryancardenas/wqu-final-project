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
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.optimize import brentq, root_scalar
from tqdm import tqdm

from wqufp.options import BSCall, BSPut
from wqufp.yield_curve import RiskFreeYield

load_dotenv()  # Load our secrets from the .env file

MAX_WORKERS = 8
DAYS_PER_YEAR = 365
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
TRG_OPT_DF_DIR = PACKAGE_ROOT / "data/opt_df"


def get_shiller_dividend_yield() -> pd.DataFrame:
    """
    Loads Robert J. Shiller's data from disk and uses it to derive SPX dividend yield.
    Source: https://shillerdata.com/

    Returns
    -------
    shiller: pd.DataFrame
        Annualized dividend yields, interpolated by Shiller to monthly intervals.
    """
    shiller_fn = PACKAGE_ROOT / "data/ie_data.xls"
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

    Also computes SPX historical daily log-returns and detrended log-returns.
    Detrending logic follows these steps:
    (1) Convert Sf series to log-returns (lr).
    (2) Compute historical mean log-return (lr_mean_historical) on expanding basis
        (window with anchored left boundary and rolling right boundary) that accounts
        for all available historical data.
    (3) Subtract the historical mean log-return from the daily log-return, producing
        the detrended log-return (lr_detrended).
    (4) Repeat for overnight log-returns (onlr) from CLOSE(T-1) to OPEN(T), as these
        are useful for reconstructing AM settlement prices.

    Returns
    -------
    spx: pd.DataFrame
        Dataframe containing DATETIME, OPEN, CLOSE, Q (annualized dividend yield), LR
        (log-return), LR_MEAN_HISTORICAL (mean historical log-return on rolling
        basis), LR_DETRENDED (LR - LR_MEAN_HISTORICAL), and the overnight
        LR-equivalents for SPX.
    """
    # Import SPX historical prices
    yf_spx_fn = PACKAGE_ROOT / "data/yf_spx.csv"
    spx = pd.read_csv(yf_spx_fn).set_index("datetime")[["open", "close"]]
    spx.index = pd.to_datetime(spx.index)

    # Join with Shiller's dividend yield
    shiller = get_shiller_dividend_yield()
    spx = spx.join(shiller, how="left")
    spx.loc[:, "q"] = spx["q"].ffill()

    # Compute log-return and detrended log-return
    spx["lr"] = np.log(spx["close"] / spx["close"].shift())
    spx["lr_mean_historical"] = spx["lr"].expanding().mean()
    spx["lr_detrended"] = spx["lr"] - spx["lr_mean_historical"]

    # Compute overnight log-return and detrended log-return
    spx["onlr"] = np.log(spx["open"] / spx["close"].shift())
    spx["onlr_mean_historical"] = spx["onlr"].expanding().mean()
    spx["onlr_detrended"] = spx["onlr"] - spx["onlr_mean_historical"]

    # Verify log-return calculation
    a = spx["close"].iloc[0] * np.exp(spx["lr"].cumsum()).values[1:]
    b = spx["close"].values[1:]
    assert np.allclose(a, b)

    spx.dropna(inplace=True)
    return spx


def load_concatenate_options_files(
    opt_fn_list: list[str],
    cols: list[str],
) -> pd.DataFrame:
    """Iterates through each filepath in opt_fn_list, loading the data and filtering
    out invalid deltas/strikes/DTE/MIDs, and then concatenates the results.

    Returns
    -------
    opt_df: pd.DataFrame
        Concatenated and filtered options data.
    """
    opt_df = pd.DataFrame()
    for i in range(len(opt_fn_list)):
        tmp = pd.read_csv(opt_fn_list[i], names=cols)

        # Ignore out-of-range deltas
        upper_delta_threshold = tmp["delta"].abs() < 1.0
        lower_delta_threshold = tmp["delta"].abs() > 0.0

        # Filter invalid strikes
        valid_strikes = tmp["strike"] > 0

        # Filter out 0 DTE options
        not_zero_dte = tmp["trade date"] != tmp["expiry date"]

        # Apply filters
        tmp = tmp[
            upper_delta_threshold & lower_delta_threshold & valid_strikes & not_zero_dte
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

    # Cast strings as datetime objects
    opt_df["trade date"] = pd.to_datetime(opt_df["trade date"])
    opt_df["expiry date"] = pd.to_datetime(opt_df["expiry date"])

    # Adjust for obsolete Saturday expiry convention
    sat_idx = opt_df[opt_df["expiry date"].dt.dayofweek == 5].index
    opt_df.loc[sat_idx, "expiry date"] = opt_df.loc[
        sat_idx, "expiry date"
    ] - pd.Timedelta(days=1)

    # Compute contract duration
    opt_df["dte"] = (opt_df["expiry date"] - opt_df["trade date"]).dt.days
    opt_df["T"] = opt_df["dte"] / DAYS_PER_YEAR

    # Identify standard monthly contracts
    is_friday = opt_df["expiry date"].dt.weekday == 4
    is_third_week = (opt_df["expiry date"].dt.day >= 15) & (
        opt_df["expiry date"].dt.day <= 21
    )
    opt_df["standard"] = is_friday & is_third_week

    # Need to filter out 0 DTE options after correcting 'expiry date' for Saturday
    # convention.
    opt_df = opt_df[opt_df["dte"] > 0]

    return opt_df


def apply_underlying_price_at_expiry(
    opt_df: pd.DataFrame, spx_data: pd.DataFrame
) -> pd.DataFrame:
    """Computes the underlying settlement price at expiry. Handles standard/non-standard
    expiry AM/PM settlements.

    Returns
    -------
    opt_df: pd.DataFrame
        Options data with Sf label included.
    """

    # Identify SPX CLOSE at expiry date
    opt_df = (
        opt_df.set_index("expiry date")
        .join(spx_data["close"].rename("Sf"), how="inner", lsuffix="", rsuffix="_s")
        .reset_index(drop=False)
    )  # Transform back to integer index for next step...

    # Apply SPX OPEN for standard (monthly, AM-settlement) options
    # Need to operate on integer index; datetime index takes too long to slice
    standard_idx = opt_df[opt_df["standard"]].index
    opt_df.loc[standard_idx, "Sf"] = (
        opt_df.set_index("expiry date")
        .join(
            spx_data["open"].rename("Sf"),
            how="left",
            lsuffix="_original",
            rsuffix="_new",
        )
        .reset_index(drop=False)
        .loc[standard_idx, "Sf_new"]
    )

    return opt_df


def compute_detrended_sf(
    market_df: pd.DataFrame,
    trade_date: str,
    expiry_date: str,
    am_settlement: bool,
    rf: Optional[float],
) -> float:
    """Computes the final price (Sf) of the underlying after detrending it.
    Detrending involves the following steps:

    PRECOMPUTED IN MARKET_DF
    (1) Convert Sf series to log-returns (lr).
    (2) Compute historical mean log-return (lr_mean_historical) on expanding basis
        (window with anchored left boundary and rolling right boundary) that accounts
        for all available historical data.
    (3) Subtract the historical mean log-return from the daily log-return, producing
        the detrended log-return (lr_detrended). If rf is provided, add it back in to
        "center" the data around the risk-free rate, which amounts to detrending by
        subtracting (lr_mean_historical - rf).

    Returns
    -------
    sf_detrended: float
        The reconstructed settlement price of the underlying, with the historical
        log-return trend removed.
    """
    x = market_df.loc[trade_date:expiry_date]  # type: ignore[misc]

    if rf is None:
        rf = 0.0

    # We ADD rf because lr_detrended = lr - lr_mean_historical, and we want
    # lr_detrended = lr - (lr_mean_historial - rf). This causes the sign of rf to
    # become positive.
    lr_detrended = x["lr_detrended"].values + rf

    # First LR is from earlier period -> set to zero.
    lr_detrended[0] = 0

    # Handle last log-return from CLOSE(T-1) to OPEN(T)
    if am_settlement:
        lr_detrended[-1] = x["onlr_detrended"].values[-1]

    # Accumulate de-trended log-returns and apply final value
    lr_final = lr_detrended.sum()
    sf_detrended: float = x.loc[trade_date, "close"] * np.exp(lr_final)
    return sf_detrended


def detrend_returns(
    opt_df: pd.DataFrame, spx_df: pd.DataFrame, rf_curve_obj: RiskFreeYield
) -> pd.DataFrame:

    date_pairs = opt_df[["trade date", "expiry date", "standard"]].drop_duplicates()

    for tup in list(date_pairs.itertuples()):
        trade_date = tup[1]
        expiry_date = tup[2]
        am_settlement = tup[3]

        ytm_offset = pd.Timedelta(hours=0)
        if am_settlement:
            ytm_offset = pd.Timedelta(hours=6.5)
        ytm = pd.to_datetime(expiry_date) - pd.to_datetime(trade_date) - ytm_offset
        rf = rf_curve_obj(date=trade_date, ytm=ytm)

        sf_detrended = compute_detrended_sf(
            market_df=spx_df,
            trade_date=trade_date,
            expiry_date=expiry_date,
            am_settlement=am_settlement,
            rf=None,
        )

        sf_centered = compute_detrended_sf(
            market_df=spx_df,
            trade_date=trade_date,
            expiry_date=expiry_date,
            am_settlement=am_settlement,
            rf=rf,
        )

        idx = opt_df[
            (opt_df["trade date"] == trade_date)
            & (opt_df["expiry date"] == expiry_date)
            & (opt_df["standard"] == am_settlement)
        ].index

        opt_df.loc[idx, "Sf_detrended"] = sf_detrended
        opt_df.loc[idx, "Sf_centered"] = sf_centered

    return opt_df


def apply_underlying_price_and_yield_at_entry(
    opt_df: pd.DataFrame,
    spx_data: pd.DataFrame,
) -> pd.DataFrame:
    """Applies the underlying price and dividend yield at 'trade date'.
    Note: For some reason, after reindexing on 'trade date' for the purposes of
    joining with spx_data, calling reset_index() causes the column to be renamed
    'index'. This does not happen with 'expiry date'. For this reason, we rename the
    'index' column back to 'trade date' after resetting the index.

    Returns
    -------
    opt_df: pd.DataFrame
        Options data with S0 (initial SPX price) and q (annualized dividend yield)
        labels included.
    """
    # Identify SPX CLOSE at trade date
    opt_df = opt_df.set_index("trade date").join(
        spx_data["close"].rename("S0"), how="left", lsuffix="", rsuffix="_s"
    )

    # Align S&P 500 dividend yield
    # Not sure why 'trade date' is special, but it requires renaming
    # from 'index' after joining with spx_data.
    opt_df = (
        opt_df.join(spx_data["q"], how="left", lsuffix="", rsuffix="_s")
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
    use_shimko: bool = False,
    worker_id: int = 0,
) -> pd.DataFrame:
    """Computes sigma (implied volatility), delta, theta, gamma, rho, and vega. Has
    'use_shimko' flag to compute using Shimko's (1993) method. If Shimko's method is
    not used, will default to using Shiller's dividend yields and FRED's treasury
    rates via wqufp.yield_curve.RiskFreeYield.

    Returns
    -------
    opt_df: pd.DataFrame
        Options data with Greeks included.
    """
    in_suffix = ""
    out_suffix = "_"
    msg_insert = ""
    if use_shimko:
        in_suffix = "_shimko"
        out_suffix = "_shimko"
        msg_insert = " via Shimko's method"

    for i in tqdm(
        range(opt_df.shape[0]),
        desc=f"Computing options Greeks{msg_insert}...",
        position=worker_id,
    ):
        row = opt_df.iloc[i]

        opt_type = row["call/put"]
        if opt_type == "c":
            greek = BSCall()
        elif opt_type == "p":
            greek = BSPut()
        else:
            raise ValueError(f"call/put should be either 'c' or 'p'; got {opt_type}")

        def f(sigma_: float) -> float:
            y_pred: float = greek.price(
                S0=row["S0"],
                K=row["strike"],
                T=row["T"],
                r=row["rf" + in_suffix],
                sigma=sigma_,
                q=row["q" + in_suffix],
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
            r=row["rf" + in_suffix],
            sigma=sigma,
            q=row["q" + in_suffix],
        )

        opt_df.loc[i, "sigma" + out_suffix] = sigma
        opt_df.loc[i, "delta" + out_suffix] = delta
        opt_df.loc[i, "theta" + out_suffix] = theta
        opt_df.loc[i, "rho" + out_suffix] = rho
        opt_df.loc[i, "gamma" + out_suffix] = gamma
        opt_df.loc[i, "vega" + out_suffix] = vega
        opt_df.loc[i, "pitm" + out_suffix] = p_itm

    return opt_df


def apply_shimkos_method(call_df: pd.DataFrame, put_df: pd.DataFrame) -> pd.DataFrame:
    """Computes options-implied carry rate and dividend yield via Shimko's (1993)
    method, which involves using put-call parity and linear regression to infer the
    carry rates and dividend yields from the following equation:
        C - P = -BK + SD
    where
        C  = call price
        P = put price
        K = strike price
        B = e^(-rT)
        S = underlying price
        D = e^(-qT)

    A line is regressed on y(K) = C - P, producing constants (-B, SD) which can then
    be used to compute r and q, respectively.

    Returns
    -------
    opt_df: pd.DataFrame
        Options data containing only those strikes for which both valid call data and
        valid put data are available. Includes three new columns: 'c-p' (difference
        between call and put price), 'rf_shimko' (implied carry rate), and 'q_shimko'
        (implied dividend yield).
    """
    assert np.array_equal(call_df.columns, put_df.columns)
    idx_labels = ["trade date", "dte", "strike"]
    call_suffix = ""
    put_suffix = "_p"
    new_cols = ["c-p", "rf_shimko", "q_shimko"]

    calls = call_df.set_index(idx_labels)
    puts = put_df.set_index(idx_labels)

    ladder = calls.join(
        puts, how="inner", lsuffix=call_suffix, rsuffix=put_suffix
    ).reset_index(drop=False)
    ladder["c-p"] = ladder["mid price" + call_suffix] - ladder["mid price" + put_suffix]

    call_cols = (
        idx_labels
        + [col + call_suffix for col in call_df.columns if col not in idx_labels]
        + new_cols
    )
    put_cols = (
        idx_labels
        + [col + put_suffix for col in call_df.columns if col not in idx_labels]
        + new_cols
    )

    for trade_date in ladder["trade date"].unique():
        tmp1 = ladder[ladder["trade date"] == trade_date]
        for dte in tmp1["dte"].unique():
            tmp2 = tmp1[tmp1["dte"] == dte]
            if tmp2.shape[0] >= 2:
                T = tmp2["T"].iloc[0]
                S0 = tmp2["S0"].iloc[0]
                K = tmp2["strike"].values
                c_minus_p = tmp2["c-p"].values

                polyfit = np.polynomial.Polynomial.fit(K, c_minus_p, deg=1)
                coefs = polyfit.convert(
                    domain=polyfit.domain, window=polyfit.domain
                ).coef
                b = coefs[0]  # Polynomial provides coefs in ascending degree
                m = coefs[1]

                rf = np.log(-m) / -T
                q = np.log(b / S0) / -T
            else:
                rf = np.nan
                q = np.nan

            idxs = tmp2.index
            ladder.loc[idxs, "rf_shimko"] = rf
            ladder.loc[idxs, "q_shimko"] = q

    calls = ladder[call_cols]
    puts = ladder[put_cols]
    puts.columns = calls.columns

    result = pd.concat([calls, puts], ignore_index=True)

    return result


def save_to_hdf5(df: pd.DataFrame, trg_path: Path | str) -> None:
    """Append df to HDF5 file, converting datetime columns to integers (seconds since
    epoch).
    """
    with h5py.File(trg_path, "a") as f:
        for col in tqdm(df.columns, desc="Appending columns to HDF5"):
            dtype = df[col].dtype
            data = df[col].values
            if dtype.kind == "M":
                data = pd.to_datetime(df[col]).astype(int).values / 1e9
                dtype = np.float64
            elif dtype.kind == "O":
                dtype = h5py.string_dtype(encoding="utf-8", length=1)

            if col not in f:
                dset = f.require_dataset(
                    name=col,
                    shape=(0,),
                    maxshape=(None,),
                    dtype=dtype,
                    exact=True,
                    compression="gzip",
                    compression_opts=9,
                )
            else:
                dset = f[col]

            n = dset.shape[0]
            dset.resize(n + data.shape[0], axis=0)
            dset[n:] = data


def rebuild_hdf5(fn_stem: str = "options_rebuilt", complevel: int = 9) -> None:
    """Helper function for rebuilding HDF5 file from batch CSV files without having
    to recompute options Greeks, which would take a long time. This is not called
    within the __main__ script, but is included here in case __main__ fails to
    aggregate the CSV files into an HDF5 file.
    """
    trg_path_hdf5 = TRG_OPT_DF_DIR / f"{fn_stem}.hdf5"
    opt_fn_list = sorted(list(TRG_OPT_DF_DIR.glob("*.csv")))
    acquired_dtypes = False
    for opt_fn in tqdm(opt_fn_list, desc="Rebuilding HDF5 via Pytables"):
        df = pd.read_csv(opt_fn)
        if not acquired_dtypes:
            dtypes = df.dtypes
            acquired_dtypes = True
        df.astype(dtypes).to_hdf(
            path_or_buf=trg_path_hdf5,
            key="options",
            mode="a",
            complevel=complevel,
            complib="zlib",
            format="table",
            append=True,
        )
    print("Processing complete!")


def preprocess_csv_pipeline(
    data_year_str: str,
    spx_ohlc_df: pd.DataFrame,
    rf_curve_obj: RiskFreeYield,
    column_names: list[str],
    options_data_dir: str,
    wid: int = 0,
) -> Path:
    start = time.perf_counter()

    # Silence common warnings that do not impede operation
    warnings.filterwarnings("ignore", message="Derivative was zero")
    warnings.filterwarnings(
        "ignore", message="divide by zero encountered in scalar divide"
    )
    warnings.filterwarnings("ignore", message="invalid value encountered in log")

    print(f"Processing year {data_year_str}...")
    opt_fn_list = sorted(glob.glob(options_data_dir + f"/SPX/*_{data_year_str}_*"))

    opt_df = load_concatenate_options_files(opt_fn_list=opt_fn_list, cols=column_names)
    opt_df = compute_contract_durations_and_types(opt_df=opt_df)
    opt_df = apply_underlying_price_at_expiry(opt_df=opt_df, spx_data=spx_ohlc_df)
    opt_df = apply_underlying_price_and_yield_at_entry(
        opt_df=opt_df, spx_data=spx_ohlc_df
    )
    opt_df = apply_risk_free_rate(opt_df=opt_df, rf_curve=rf_curve_obj)

    call_df = process_call_options(opt_df=opt_df)
    put_df = process_put_options(opt_df=opt_df)

    # Apply Shimko's method for calculating rf and q
    opt_df = apply_shimkos_method(call_df=call_df, put_df=put_df)

    # Compute Greeks using Shiller/FRED assumptions
    opt_df = compute_options_greeks(opt_df=opt_df, use_shimko=False, worker_id=wid)

    # Compute Greeks using Shimko assumptions
    opt_df = compute_options_greeks(opt_df=opt_df, use_shimko=True, worker_id=wid)

    # Compute detrended and centered final underlying prices
    opt_df = detrend_returns(opt_df=opt_df, spx_df=spx_df, rf_curve_obj=rf_curve_obj)

    # Save to disk
    trg_path_csv = TRG_OPT_DF_DIR / (data_year_str + ".csv")
    opt_df.to_csv(trg_path_csv, index=False)

    stop = time.perf_counter()
    print(
        f"Processed data from year {data_year_str} in {(stop - start) / 60:.2f} "
        f"minutes!"
    )
    return trg_path_csv


if __name__ == "__main__":
    print("Initiating...")
    trg_path_hdf5 = TRG_OPT_DF_DIR / "options.hdf5"
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
    data_years = list(pd.Series([c.split("_")[1] for c in all_opt_fn_list]).unique())

    pipeline = partial(
        preprocess_csv_pipeline,
        spx_ohlc_df=spx_df,
        rf_curve_obj=rf_curve,
        column_names=cols,
        options_data_dir=options_dir,
    )
    # Parallelize preprocessing pipeline
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [
            pool.submit(pipeline, data_year, wid=wid)
            for wid, data_year in enumerate(data_years)
        ]
        csv_list = [f.result() for f in futures]

    print("Finished parallel execution!")
    print("*" * 50)
    print("Aggregating CSV files...")
    # Aggregate CSV files into single HDF5 file
    for csv_fn in csv_list:
        options_df = pd.read_csv(csv_fn)

        # Append columns to HDF5
        save_to_hdf5(df=options_df, trg_path=trg_path_hdf5)

    print("*" * 50)
    print("Process Complete!")
