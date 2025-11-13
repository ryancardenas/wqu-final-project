#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
FILE: test_options.py
PROJECT: wqu-final-project
ORIGINAL AUTHOR: ryancardenas
DATE CREATED: 11/12/25

Tests for methods contained in options.py.
"""
# mypy: ignore-errors

import numpy as np
import pytest

import wqufp.options as op


class Test_BSCall_greeks:
    """Tests that BSCall.greeks() meets the following expectations:

    Delta:
        - Is greater than or equal to zero.
        - Is less than or equal to exp(-qT).
        - Increases monotonically with S0.
    """

    @pytest.mark.parametrize("S0", 10 ** np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    @pytest.mark.parametrize("K_mult", np.r_[np.linspace(0.7, 1.3, 4), np.array([1.0])])
    @pytest.mark.parametrize("T", np.linspace(1 / 365, 2.0, 5))
    @pytest.mark.parametrize("r", np.linspace(0.0, 0.09, 3))
    @pytest.mark.parametrize("sigma", np.linspace(0.01, 0.3, 5))
    @pytest.mark.parametrize("q", np.linspace(0.0, 0.03, 4))
    def test_delta_gte_zero(self, S0, K_mult, T, r, sigma, q):
        K = S0 * K_mult
        call = op.BSCall()
        delta, theta, rho, gamma, vega = call.greeks(
            S0=S0, K=K, T=T, r=r, sigma=sigma, q=q
        )
        assert 0 <= delta

    @pytest.mark.parametrize("S0", 10 ** np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    @pytest.mark.parametrize("K_mult", np.r_[np.linspace(0.7, 1.3, 4), np.array([1.0])])
    @pytest.mark.parametrize("T", np.linspace(1 / 365, 2.0, 5))
    @pytest.mark.parametrize("r", np.linspace(0.0, 0.09, 3))
    @pytest.mark.parametrize("sigma", np.linspace(0.01, 0.3, 5))
    @pytest.mark.parametrize("q", np.linspace(0.0, 0.03, 4))
    def test_delta_lte_discounted_dividend(self, S0, K_mult, T, r, sigma, q):
        K = S0 * K_mult
        call = op.BSCall()
        delta, theta, rho, gamma, vega = call.greeks(
            S0=S0, K=K, T=T, r=r, sigma=sigma, q=q
        )
        assert delta <= np.exp(-q * T)

    @pytest.mark.parametrize("S0", [np.arange(0.0, 200.0, 0.3)])
    @pytest.mark.parametrize("K", [100])
    @pytest.mark.parametrize("T", np.linspace(1 / 365, 2.0, 3))
    @pytest.mark.parametrize("r", np.linspace(0.0, 0.09, 3))
    @pytest.mark.parametrize("sigma", np.linspace(0.01, 0.3, 3))
    @pytest.mark.parametrize("q", np.linspace(0.0, 0.03, 3))
    def test_delta_increases_monotonically_with_S0(self, S0, K, T, r, sigma, q):
        call = op.BSCall()
        delta_vec = np.full_like(S0, fill_value=np.nan)
        for i in range(S0.shape[0]):
            delta_vec[i], theta, rho, gamma, vega = call.greeks(
                S0=S0[i], K=K, T=T, r=r, sigma=sigma, q=q
            )
        assert np.all(delta_vec[1:] >= delta_vec[:-1])


class Test_BSCPut_greeks:
    """Tests that BSPut.greeks() meets the following expectations:

    Delta:
        - Is less than or equal to zero.
        - Is greater than or equal to -exp(-qT).
        - Increases monotonically with S0.
    """

    @pytest.mark.parametrize("S0", 10 ** np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    @pytest.mark.parametrize("K_mult", np.r_[np.linspace(0.7, 1.3, 4), np.array([1.0])])
    @pytest.mark.parametrize("T", np.linspace(1 / 365, 2.0, 5))
    @pytest.mark.parametrize("r", np.linspace(0.0, 0.09, 3))
    @pytest.mark.parametrize("sigma", np.linspace(0.01, 0.3, 5))
    @pytest.mark.parametrize("q", np.linspace(0.0, 0.03, 4))
    def test_delta_lte_zero(self, S0, K_mult, T, r, sigma, q):
        K = S0 * K_mult
        put = op.BSPut()
        delta, theta, rho, gamma, vega = put.greeks(
            S0=S0, K=K, T=T, r=r, sigma=sigma, q=q
        )
        assert 0 >= delta

    @pytest.mark.parametrize("S0", 10 ** np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    @pytest.mark.parametrize("K_mult", np.r_[np.linspace(0.7, 1.3, 4), np.array([1.0])])
    @pytest.mark.parametrize("T", np.linspace(1 / 365, 2.0, 5))
    @pytest.mark.parametrize("r", np.linspace(0.0, 0.09, 3))
    @pytest.mark.parametrize("sigma", np.linspace(0.01, 0.3, 5))
    @pytest.mark.parametrize("q", np.linspace(0.0, 0.03, 4))
    def test_delta_gte_discounted_dividend(self, S0, K_mult, T, r, sigma, q):
        K = S0 * K_mult
        put = op.BSPut()
        delta, theta, rho, gamma, vega = put.greeks(
            S0=S0, K=K, T=T, r=r, sigma=sigma, q=q
        )
        assert delta >= -np.exp(-q * T)

    @pytest.mark.parametrize("S0", [np.arange(0.0, 200.0, 0.3)])
    @pytest.mark.parametrize("K", [100])
    @pytest.mark.parametrize("T", np.linspace(1 / 365, 2.0, 3))
    @pytest.mark.parametrize("r", np.linspace(0.0, 0.09, 3))
    @pytest.mark.parametrize("sigma", np.linspace(0.01, 0.3, 3))
    @pytest.mark.parametrize("q", np.linspace(0.0, 0.03, 3))
    def test_delta_increases_monotonically_with_S0(self, S0, K, T, r, sigma, q):
        put = op.BSPut()
        delta_vec = np.full_like(S0, fill_value=np.nan)
        for i in range(S0.shape[0]):
            delta_vec[i], theta, rho, gamma, vega = put.greeks(
                S0=S0[i], K=K, T=T, r=r, sigma=sigma, q=q
            )
        assert np.all(delta_vec[1:] >= delta_vec[:-1])
