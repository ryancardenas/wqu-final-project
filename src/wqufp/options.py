#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
FILE: options.py
PROJECT: wqu-final-project
ORIGINAL AUTHOR: ryancardenas
DATE CREATED: 11/12/25

Code for computing options prices and Greeks.
"""

from typing import Union

import numpy as np
from scipy.stats import norm

Numeric = Union[int, float]


class BSCall:
    def price(
        self,
        S0: Numeric,
        K: Numeric,
        T: Numeric,
        r: Numeric,
        sigma: Numeric,
        q: Numeric = 0,
    ) -> Numeric:
        """Computes Black-Scholes call price for European options.

        Parameters
        ----------
        S0 : int or float
            Current underlying price.

        K: int or float
            Option strike price.

        T: int or float
            Time to maturity, measured in years.

        r: int or float
            Risk-free rate or cost-of-carry, expressed as a decimal (example: "0.045"
            means 4.5%).

        sigma: int or float
            Volatility parameter.

        q: int or float
            Annualized continuous dividend yield.

        Returns
        -------
        C : float
            The price of the European call option.
        """
        assert T > 0, "T must be greater than 0"

        vol = sigma * np.sqrt(T)
        d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / vol
        d2 = d1 - vol

        C: Numeric = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(
            d2
        )
        return C

    def greeks(
        self,
        S0: Numeric,
        K: Numeric,
        T: Numeric,
        r: Numeric,
        sigma: Numeric,
        q: Numeric = 0,
    ) -> tuple[Numeric, Numeric, Numeric, Numeric, Numeric]:
        """Computes Black-Scholes call greeks for European options.

        Parameters
        ----------
        S0 : int or float
            Current underlying price.

        K: int or float
            Option strike price.

        T: int or float
            Time to maturity, measured in years.

        r: int or float
            Risk-free rate or cost-of-carry, expressed as a decimal (example: "0.045"
            means 4.5%).

        sigma: int or float
            Volatility parameter.

        q: int or float
            Annualized continuous dividend yield.

        Returns
        -------
        (c_delta, c_theta, c_rho, gamma, vega) : tuple of floats
            The delta, theta, rho, gamma, and vega of the call option.
        """
        assert T > 0, "T must be greater than 0"

        vol = sigma * np.sqrt(T)
        d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / vol
        d2 = d1 - vol

        c_delta = np.exp(-q * T) * norm.cdf(d1)
        c_theta = (
            -S0 * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
            + q * S0 * np.exp(-q * T) * norm.cdf(d1)
        )
        c_rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        gamma = np.exp(-q * T) * norm.pdf(d1) / (S0 * vol)
        vega = S0 * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

        return (c_delta, c_theta, c_rho, gamma, vega)

    def delta(
        self,
        S0: Numeric,
        K: Numeric,
        T: Numeric,
        r: Numeric,
        sigma: Numeric,
        q: Numeric = 0,
    ) -> Numeric:
        """Computes Black-Scholes call delta for European options."""
        assert T > 0, "T must be greater than 0"

        vol = sigma * np.sqrt(T)
        d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / vol

        c_delta: Numeric = np.exp(-q * T) * norm.cdf(d1)
        return c_delta

    def theta(
        self,
        S0: Numeric,
        K: Numeric,
        T: Numeric,
        r: Numeric,
        sigma: Numeric,
        q: Numeric = 0,
    ) -> Numeric:
        """Computes Black-Scholes call theta for European options."""
        assert T > 0, "T must be greater than 0"

        vol = sigma * np.sqrt(T)
        d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / vol
        d2 = d1 - vol

        c_theta: Numeric = (
            -S0 * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
            + q * S0 * np.exp(-q * T) * norm.cdf(d1)
        )
        return c_theta

    def rho(
        self,
        S0: Numeric,
        K: Numeric,
        T: Numeric,
        r: Numeric,
        sigma: Numeric,
        q: Numeric = 0,
    ) -> Numeric:
        """Computes Black-Scholes call rho for European options."""
        assert T > 0, "T must be greater than 0"

        vol = sigma * np.sqrt(T)
        d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / vol
        d2 = d1 - vol

        c_rho: Numeric = K * T * np.exp(-r * T) * norm.cdf(d2)
        return c_rho

    def gamma(
        self,
        S0: Numeric,
        K: Numeric,
        T: Numeric,
        r: Numeric,
        sigma: Numeric,
        q: Numeric = 0,
    ) -> Numeric:
        """Computes Black-Scholes call gamma for European options."""
        assert T > 0, "T must be greater than 0"

        vol = sigma * np.sqrt(T)
        d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / vol

        gamma: Numeric = np.exp(-q * T) * norm.pdf(d1) / (S0 * vol)
        return gamma

    def vega(
        self,
        S0: Numeric,
        K: Numeric,
        T: Numeric,
        r: Numeric,
        sigma: Numeric,
        q: Numeric = 0,
    ) -> Numeric:
        """Computes Black-Scholes call vega for European options."""
        assert T > 0, "T must be greater than 0"

        vol = sigma * np.sqrt(T)
        d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / vol

        vega: Numeric = S0 * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
        return vega


class BSPut:
    def price(
        self,
        S0: Numeric,
        K: Numeric,
        T: Numeric,
        r: Numeric,
        sigma: Numeric,
        q: Numeric = 0,
    ) -> Numeric:
        """Computes Black-Scholes put price for European options.

        Parameters
        ----------
        S0 : int or float
            Current underlying price.

        K: int or float
            Option strike price.

        T: int or float
            Time to maturity, measured in years.

        r: int or float
            Risk-free rate or cost-of-carry, expressed as a decimal (example: "0.045"
            means 4.5%).

        sigma: int or float
            Volatility parameter.

        q: int or float
            Annualized continuous dividend yield.

        Returns
        -------
        P : float
            The price of the European call option.
        """
        assert T > 0, "T must be greater than 0"

        vol = sigma * np.sqrt(T)
        d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / vol
        d2 = d1 - vol

        P: Numeric = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(
            -q * T
        ) * norm.cdf(-d1)
        return P

    def greeks(
        self,
        S0: Numeric,
        K: Numeric,
        T: Numeric,
        r: Numeric,
        sigma: Numeric,
        q: Numeric = 0,
    ) -> tuple[Numeric, Numeric, Numeric, Numeric, Numeric]:
        """Computes Black-Scholes put greeks for European options.

        Parameters
        ----------
        S0 : int or float
            Current underlying price.

        K: int or float
            Option strike price.

        T: int or float
            Time to maturity, measured in years.

        r: int or float
            Risk-free rate or cost-of-carry, expressed as a decimal (example: "0.045"
            means 4.5%).

        sigma: int or float
            Volatility parameter.

        q: int or float
            Annualized continuous dividend yield.

        Returns
        -------
        (p_delta, p_theta, p_rho, gamma, vega) : tuple of floats
            The delta, theta, rho, gamma, and vega of the put option.
        """
        assert T > 0, "T must be greater than 0"

        vol = sigma * np.sqrt(T)
        d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / vol
        d2 = d1 - vol

        p_delta = -np.exp(-q * T) * norm.cdf(-d1)
        p_theta = (
            -S0 * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
            - q * S0 * np.exp(-q * T) * norm.cdf(-d1)
        )
        p_rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        gamma = np.exp(-q * T) * norm.pdf(d1) / (S0 * vol)
        vega = S0 * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

        return (p_delta, p_theta, p_rho, gamma, vega)

    def delta(
        self,
        S0: Numeric,
        K: Numeric,
        T: Numeric,
        r: Numeric,
        sigma: Numeric,
        q: Numeric = 0,
    ) -> Numeric:
        """Computes Black-Scholes put delta for European options."""
        assert T > 0, "T must be greater than 0"

        vol = sigma * np.sqrt(T)
        d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / vol

        p_delta: Numeric = -np.exp(-q * T) * norm.cdf(-d1)
        return p_delta

    def theta(
        self,
        S0: Numeric,
        K: Numeric,
        T: Numeric,
        r: Numeric,
        sigma: Numeric,
        q: Numeric = 0,
    ) -> Numeric:
        """Computes Black-Scholes put theta for European options."""
        assert T > 0, "T must be greater than 0"

        vol = sigma * np.sqrt(T)
        d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / vol
        d2 = d1 - vol

        p_theta: Numeric = (
            -S0 * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
            - q * S0 * np.exp(-q * T) * norm.cdf(-d1)
        )
        return p_theta

    def rho(
        self,
        S0: Numeric,
        K: Numeric,
        T: Numeric,
        r: Numeric,
        sigma: Numeric,
        q: Numeric = 0,
    ) -> Numeric:
        """Computes Black-Scholes put rho for European options."""
        assert T > 0, "T must be greater than 0"

        vol = sigma * np.sqrt(T)
        d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / vol
        d2 = d1 - vol

        p_rho: Numeric = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        return p_rho

    def gamma(
        self,
        S0: Numeric,
        K: Numeric,
        T: Numeric,
        r: Numeric,
        sigma: Numeric,
        q: Numeric = 0,
    ) -> Numeric:
        """Computes Black-Scholes put gamma for European options."""
        assert T > 0, "T must be greater than 0"

        vol = sigma * np.sqrt(T)
        d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / vol

        gamma: Numeric = np.exp(-q * T) * norm.pdf(d1) / (S0 * vol)
        return gamma

    def vega(
        self,
        S0: Numeric,
        K: Numeric,
        T: Numeric,
        r: Numeric,
        sigma: Numeric,
        q: Numeric = 0,
    ) -> Numeric:
        """Computes Black-Scholes put vega for European options."""
        assert T > 0, "T must be greater than 0"

        vol = sigma * np.sqrt(T)
        d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / vol

        vega: Numeric = S0 * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
        return vega
