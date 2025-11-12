#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
FILE: yield_curve.py
PROJECT: wqu-final-project
ORIGINAL AUTHOR: ryancardenas
DATE CREATED: 11/11/25

Tooling for modeling risk-free interest rate as a function of trade date and
time-to-maturity.
"""
import datetime
import importlib.resources as res
from typing import Union

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

Datelike = Union[pd._libs.tslibs.timestamps.Timestamp, datetime.date, str]
Numbers = Union[float, int, np.ndarray, np.floating, np.integer, pd.Series]


class RiskFreeYield:
    """Uses historical daily Market Yield on US Treasury Securities (Investment
    Basis)  downloaded from FRED API to interpolate risk-free rate over a continuous
    years-to-maturity domain for any day within the historical timeline.
    """

    def __init__(self) -> None:
        self._data = self._import_yield_data()
        self._T_vec = [
            0.0,
            1 / 12,
            3 / 12,
            6 / 12,
            1.0,
            2.0,
            3.0,
            5.0,
            7.0,
            10.0,
            20.0,
            30.0,
        ]

    def _import_yield_data(self) -> pd.DataFrame:
        with res.files("data").joinpath("yield_data.csv").open("r") as f:
            df = pd.read_csv(f, index_col=0)
            df.index = pd.to_datetime(df.index)
        return df

    def __call__(self, date: Datelike, ytm: Numbers) -> Numbers:
        """Returns estimated risk-free rate at 'date' for 'ytm' years to maturity."""
        assert (
            pd.to_datetime(date) >= self._data.index[0]
        ), f"dates earlier than {self._data.index[0]} not allowed"
        assert (
            pd.to_datetime(date) <= self._data.index[-1]
        ), f"dates later than {self._data.index[-1]} not allowed"

        if pd.to_datetime(date) not in self._data.index:
            datetime = self._data.loc[: pd.to_datetime(date)].index[-1]
        else:
            datetime = pd.to_datetime(date)
        y = self._data.loc[datetime]
        cspline = CubicSpline(x=self._T_vec, y=y, bc_type=("not-a-knot", "clamped"))
        return cspline(ytm)
