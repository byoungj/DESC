"""Tests of DESC against exact non-axisymmetric MHD equilibria, without solving.

``fit_exact_solution`` fits the closed form equilibria of analytic_equilibria.py on
DESC's basis, so the force balance, field, current density, enclosed current and
magnetic axis DESC computes from it must match the closed forms to the truncation
error of its basis, which falls spectrally with resolution.
"""

import numpy as np
import pytest

from .analytic_equilibria import (
    Iota2Equilibrium,
    ShearedIotaEquilibrium,
    fit_exact_solution,
    surface_errors,
    volume_errors,
)

# about 3 times the errors at L = M = N = 6, with and without stellarator symmetry
TOLERANCES = {
    Iota2Equilibrium: {"F": 5e-3, "B": 1e-4, "J": 2e-3, "current": 1e-4, "axis": 5e-8},
    ShearedIotaEquilibrium: {
        "F": 2e-4,
        "B": 3e-6,
        "J": 3e-5,
        "current": 2e-6,
        "axis": 2e-8,
    },
}


class TestExactSolutionFit:
    """Exact equilibria fitted on DESC's basis, never solved."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "exact",
        [
            Iota2Equilibrium(),
            Iota2Equilibrium(tau0=0.5),
            ShearedIotaEquilibrium(),
            ShearedIotaEquilibrium(tau1=0.15),
        ],
        ids=["iota2", "iota2 asym", "sheared", "sheared asym"],
    )
    def test_fit_matches_exact_solution(self, exact):
        """Test DESC's force balance, B, J, current and axis on exact equilibria."""
        eq = fit_exact_solution(exact, 6, 6, 6)
        assert eq.sym == exact.sym
        errors = volume_errors(exact, eq)
        for key, tol in TOLERANCES[type(exact)].items():
            assert errors[key] < tol, (key, errors[key])

    @pytest.mark.regression
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "exact",
        [Iota2Equilibrium(), ShearedIotaEquilibrium()],
        ids=["iota2", "sheared"],
    )
    def test_fit_converges_spectrally(self, exact):
        """Test errors fall spectrally, B ahead of J and F, and hold to the axis."""
        eq = fit_exact_solution(exact, 6, 6, 6)
        low = volume_errors(exact, eq)
        high = volume_errors(exact, fit_exact_solution(exact, 12, 12, 12))
        # doubling the resolution reduces every error over 1000 fold
        for key in low:
            assert high[key] < 1e-3 * low[key], (key, low[key], high[key])
        for errors in (low, high):
            assert errors["B"] < errors["J"] and errors["B"] < errors["F"]
        assert high["axis"] < 1e-12  # m
        # surfaces close to the axis are as accurate as the bulk, so the enclosed
        # current converges to its limit I / rho^2 at the axis
        near_axis = surface_errors(exact, eq, rho=np.array([0.005, 0.01, 0.02]))
        for key, errors in near_axis.items():
            assert np.all(errors < low[key]), (key, errors, low[key])
