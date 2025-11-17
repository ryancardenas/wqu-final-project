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

import matplotlib.pyplot as plt
import numpy as np

from wqufp.yield_curve import RiskFreeYield

# import pandas as pd


plt.rcParams.update({"font.size": 12})

IMG_PATH = Path("./paper/images")


def generate_treasury_yield_curve_graphs(date: str = "2025-11-01") -> None:
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

    print("Yield curve graphs saved to disk:")
    print(fn)
    print(fn2)
