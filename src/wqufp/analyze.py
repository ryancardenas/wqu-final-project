#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
FILE: analyze.py
PROJECT: wqu-final-project
ORIGINAL AUTHOR: ryancardenas
DATE CREATED: 11/17/25

Scripts for analyzing options data and generating charts.
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wqufp.yield_curve import RiskFreeYield

plt.rcParams.update({"font.size": 12})

IMG_PATH = Path("./paper/images")
HDF5_PATH = Path("./src/data/opt_df/options.hdf5")


def generate_treasury_yield_curve_graphs(date: str = "2025-11-01") -> None:
    """Generates graphs of interpolated treasury yield curve at 'date' with raw FRED
    data points displayed on top of cubic spline fit. Saves two graphs to disk -- one
    for the 30-year view and one for the 1-year view.
    """
    rf = RiskFreeYield()

    # First the 0-30 year chart
    fn = IMG_PATH / f"yield_curve_30y_on_{date}.png"
    ytm = np.linspace(0, 30.1, 1000)

    y = rf(date, ytm=ytm)

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(ytm, y, color="C0", label="cubic spline")

    idx = rf._T_vec <= ytm.max()
    ax.scatter(
        np.array(rf._T_vec)[idx],
        rf._data.loc[:date].iloc[-1].values[idx],  # type: ignore[misc]
        color="C1",
        label="raw FRED data",
    )
    ax.set_xlim(0, ytm.max())
    ax.set_xlabel("Years to Maturity")
    ax.set_ylabel("Treasury Yield [%]")
    ax.set_title(f"Treasury Yield Curve on {date}\n0-30 Years")
    ax.grid()
    ax.legend()
    plt.savefig(fn)

    # Now the 0-1 year chart
    ytm = np.linspace(0, 1.01, 1000)
    fn2 = IMG_PATH / f"yield_curve_1y_on_{date}.png"

    y = rf(date, ytm=ytm)

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(ytm, y, color="C0", label="cubic spline")

    idx = rf._T_vec <= ytm.max()
    ax.scatter(
        np.array(rf._T_vec)[idx],
        rf._data.loc[:date].iloc[-1].values[idx],  # type: ignore[misc]
        color="C1",
        label="raw FRED data",
    )
    ax.set_xlim(0, ytm.max())
    ax.set_xlabel("Years to Maturity")
    ax.set_ylabel("Treasury Yield [%]")
    ax.set_title(f"Treasury Yield Curve on {date}\n0-1 Year")
    ax.grid()
    ax.legend()
    plt.savefig(fn2)

    print("Yield curve graphs saved to disk.")
    print(fn)
    print(fn2)


def get_measures(
    dataframe: pd.DataFrame,
    q_measure: str,
    contract_type: str = "both",
    prec: int = 2,
    detrended: bool = False,
) -> pd.DataFrame:
    """Generates a dataframe containing information about the computed Q- and
    P-measures.

    Parameters:
    -------
    dataframe: pd.DataFrame
        Data loaded into memory from options.hdf5.

    q_measure: str,
        String representing the column to use as the Q-measure. Should be either
        "pitm_" for the FRED/Shiller Prob(ITM), or "pitm_shimko" for the Shimko Prob(
        ITM).

    contract_type: str
        Can be either 'c' for calls, 'p' for puts, or 'both' for all contracts.

    prec: int
        Precision with which to round Q-measure.

    detrended: bool
        Flag for using either ITM or ITM_detrended in computing Q-P.

    Returns
    -------
    result: pd.DataFrame
    """

    assert contract_type in ["c", "p", "both"]

    suffix = ""
    if detrended:
        suffix = "_detrended"

    cols = ["call/put", q_measure, "ITM" + suffix]
    if contract_type == "both":
        tmp = dataframe[cols].copy()
    else:
        tmp = dataframe[dataframe["call/put"] == contract_type][cols].copy()
    tmp = (
        tmp[[q_measure, "ITM" + suffix]]
        .sort_values(by=q_measure)
        .reset_index(drop=True)
    )
    tmp[q_measure] = tmp[q_measure].round(prec)
    grp = tmp.groupby(q_measure)

    result = grp.mean().reset_index()
    result["N"] = grp.count().values
    result.rename({q_measure: "Q", "ITM" + suffix: "P"}, axis=1, inplace=True)
    result["Q-P"] = result["Q"] - result["P"]
    return result


def load_preprocessed_data() -> pd.DataFrame:
    """Loads preprocessed options data from HDF5 file.
    Returns
    -------
    df: pd.DataFrame
        Dataframe containing preprocessed options data. Includes a just-in-time
        calculation of ITM_detrended.
    """
    date_labels = ["trade date", "expiry date"]
    val_labels = [
        "call/put",
        "standard",
        "strike",
        "S0",
        "Sf",
        "Sf_detrended",
        "ITM",
        "q",
        "rf",
        "sigma_",
        "delta_",
        "pitm_",
        "q_shimko",
        "rf_shimko",
        "sigma_shimko",
        "delta_shimko",
        "pitm_shimko",
    ]
    df = pd.DataFrame()
    with h5py.File(HDF5_PATH, "r") as f:
        print(f.keys())
        for label in date_labels:
            df[label] = pd.to_datetime(f[label], unit="s")
        for label in val_labels:
            if "/" in label:
                parts = label.split("/")
                df[label] = pd.Series(f[parts[0]][parts[1]]).str.decode("utf-8")
            else:
                df[label] = f[label]

    # Compute ITM condition for detrended Sf
    df["ITM_detrended"] = np.where(
        df["call/put"] == "c",
        df["Sf_detrended"] > df["strike"],
        df["Sf_detrended"] < df["strike"],
    )

    df.dropna(inplace=True)
    print(f"{df.memory_usage().sum() / 1e9:.2f} GB loaded into memory")
    return df


def generate_qp_graphs(
    df: pd.DataFrame, q_measure: str = "pitm_", prec: int = 2, detrended: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generates plots of Q-P and N_samples, then saves to disk.

    Parameters
    -------
    df: pd.DataFrame
        Options data loaded from HDF5 file.

    q_measure: {'pitm_', 'pitm_shimko'}, optional
        Column in df from which to source the Q-measure Prob(ITM). The value 'pitm_'
        is derived from FRED/Shimko data for rf and q; the 'pitm_shimko' data is
        derived from Shimko's linear regression method on C - P.

    prec: int, optional
        The precision with which to round all Q-measures.

    detrended: bool, optional
        Flag for using detrended Sf and ITM. Defaults to False, which uses true
        historical Sf and ITM instead.

    Returns
    -------
    (x, calls, puts): tuple
        x - Measure dataframe for all contracts.
        calls - Measure dataframe for calls only.
        puts - Measure dataframe for puts only.
    """

    if q_measure == "pitm_":
        method = "FRED/Shiller"
    elif q_measure == "pitm_shimko":
        method = "Shimko"
    else:
        raise ValueError(f"{q_measure} has no implemented method")

    detrend_label = ""
    if detrended:
        detrend_label = "Detrended"

    x_label = "Q"
    y_label = "Q-P"
    n_label = "N"
    subtitle = f"\n{method} Method, 2010-2024"

    x = get_measures(
        dataframe=df,
        q_measure=q_measure,
        contract_type="both",
        prec=prec,
        detrended=detrended,
    )
    calls = get_measures(
        dataframe=df,
        q_measure=q_measure,
        contract_type="c",
        prec=prec,
        detrended=detrended,
    )
    puts = get_measures(
        dataframe=df,
        q_measure=q_measure,
        contract_type="p",
        prec=prec,
        detrended=detrended,
    )

    # Generate graph of Q-P
    fn = (
        IMG_PATH / f"{method.lower().replace('/', '_')}_q_minus_p_"
        f"{detrend_label.lower()}.png"
    )
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_title(
        f"Differences Between Q- and {detrend_label} P-Measures for Prob(ITM)"
        + subtitle
    )
    ax.plot(x[x_label], x[y_label], label=f"{y_label}, All Contracts", color="C0")
    ax.plot(x[x_label], calls[y_label], label=f"{y_label}, Calls Only", color="C1")
    ax.plot(x[x_label], puts[y_label], label=f"{y_label}, Puts Only", color="C4")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.fill_between(
        x[x_label],
        x[y_label],
        0,
        where=(x[y_label] >= 0),
        color="green",
        alpha=0.3,
        label="Q overestimates P",
    )
    ax.fill_between(
        x[x_label],
        x[y_label],
        0,
        where=(x[y_label] < 0),
        color="red",
        alpha=0.3,
        label="Q underestimates P",
    )
    ax.axhline(color="grey")
    ax.axvline(0.5, color="grey")
    ax.grid()
    ax.legend()
    plt.savefig(fn)

    # Generate graph of N
    fn2 = IMG_PATH / f"{method.lower().replace('/', '_')}_n_{detrend_label.lower()}.png"
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_title(
        f"Number of {detrend_label} Contracts Sampled at Each Q-Measure"
        + subtitle
        + "\n"
        + r"$N_{total}=$"
        + f"{df.shape[0]:,}"
    )
    ax.plot(x[x_label], x[n_label], "-o", label=f"{y_label}, All Contracts", color="C0")
    ax.plot(x[x_label], calls[n_label], label=f"{y_label}, Calls Only", color="C1")
    ax.plot(x[x_label], puts[n_label], label=f"{y_label}, Puts Only", color="C4")
    ax.set_xlabel(x_label)
    ax.set_ylabel(n_label)
    ax.grid()
    ax.legend()
    plt.savefig(fn2)

    print(f"{method} graphs saved to disk.")
    return x, calls, puts


def generate_method_comparison_graphs(
    x: pd.DataFrame,
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    x_shimko: pd.DataFrame,
    calls_shimko: pd.DataFrame,
    puts_shimko: pd.DataFrame,
) -> None:
    """Generates graphs of Q_Shimko minus Q_FRED_Shiller and saves to disk.

    Parameters
    -------
    x: pd.DataFrame
        FRED/Shiller all contracts dataframe from get_measures().

    calls: pd.DataFrame
        FRED/Shiller calls-only dataframe from get_measures().

    puts: pd.DataFrame
        FRED/Shiller puts-only dataframe from get_measures().

    x_shimko: pd.DataFrame
        Shimko all contracts dataframe from get_measures().

    calls_shimko: pd.DataFrame
        Shimko calls-only dataframe from get_measures().

    puts_shimko: pd.DataFrame
        Shimko puts-only dataframe from get_measures().
    """
    x_label = "Q(ITM)"
    y_label = "Q"
    y_axis_label = r"$Q_{\text{Shimko}}-Q{\text{FRED_Shiller}}"
    fn = IMG_PATH / "shimko_minus_fred_shiller_q.png"

    y_lim = (-0.04, 0.02)
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))
    fig.suptitle("Comparing Shimko Q Minus FRED/Shiller Q")

    ax = axs[0]
    y = x_shimko[y_label] - x[y_label]
    ax.set_title("All Contracts")
    ax.plot(x[x_label], y, label=f"{y_label}: Shimko - FRED/Shiller")
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.set_ylabel(y_axis_label)
    ax.axhline(color="grey")
    ax.axvline(0.5, color="grey")
    ax.fill_between(
        x[x_label],
        y,
        0,
        where=(y >= 0),
        color="green",
        alpha=0.3,
        label="Shimko overestimates FRED/Shiller",
    )
    ax.fill_between(
        x[x_label],
        y,
        0,
        where=(y < 0),
        color="red",
        alpha=0.3,
        label="Shimko underestimates FRED/Shiller",
    )
    ax.set_ylim(y_lim)
    ax.grid()
    ax.legend()

    ax = axs[1]
    y = calls_shimko[y_label] - calls[y_label]
    ax.set_title("Calls Only")
    ax.plot(x[x_label], y)
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.set_ylabel(y_axis_label)
    ax.axhline(color="grey")
    ax.axvline(0.5, color="grey")
    ax.grid()
    ax.fill_between(x[x_label], y, 0, where=(y >= 0), color="green", alpha=0.3)
    ax.fill_between(x[x_label], y, 0, where=(y < 0), color="red", alpha=0.3)
    ax.set_ylim(y_lim)

    ax = axs[2]
    y = puts_shimko[y_label] - puts[y_label]
    ax.set_title("Puts Only")
    ax.plot(x[x_label], y)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_axis_label)
    ax.axhline(color="grey")
    ax.axvline(0.5, color="grey")
    ax.grid()
    ax.fill_between(x[x_label], y, 0, where=(y >= 0), color="green", alpha=0.3)
    ax.fill_between(x[x_label], y, 0, where=(y < 0), color="red", alpha=0.3)
    ax.set_ylim(y_lim)

    fig.tight_layout()
    plt.savefig(fn)
