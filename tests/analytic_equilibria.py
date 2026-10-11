"""Exact non-axisymmetric MHD equilibria, to verify DESC against closed forms.

Landreman, arXiv:2609.26742, gives two families of smooth 3D equilibria with nested
flux surfaces, finite pressure and no continuous symmetry, whose field, flux label and
pressure are elementary functions of the Cartesian position: one with rotational
transform iota = 2 on every surface (his section 2) and one with sheared iota (his
section 3). Issan, Citrin, Landreman & Tracey, arXiv:2610.07304, add to each family a
parameter that breaks stellarator symmetry, tau0 and tau1 (their sections 2 and 3).
Both families have two field periods. The papers set mu_0 = 1; here lengths are in m
and B in T, so pressure and current density carry a factor 1 / mu_0.

``fit_exact_solution`` fits an Equilibrium to the exact solution with DESC's
``set_initial_guess`` and never solves it, so the errors of the quantities DESC computes
from it are the truncation errors of its basis, with no solver involved.
"""

import functools

import numpy as np
from numpy.polynomial import Chebyshev, Polynomial
from scipy.constants import mu_0

from desc.backend import jax, jnp
from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid, QuadratureGrid
from desc.integrals import surface_averages
from desc.profiles import PowerSeriesProfile
from desc.utils import dot, rpz2xyz, rpz2xyz_vec, xyz2rpz_vec


class _ExactEquilibrium:
    """Closed form field and flux surfaces, in DESC's coordinates rho, theta, phi."""

    NFP = 2

    def axis(self, phi):
        """R, Z (m) of the magnetic axis in the plane phi."""
        return self.section(0 * phi, 0 * phi, phi)

    @functools.cached_property
    def _B_and_grad(self):
        return jax.jit(jax.vmap(lambda x: (self.B(x), jax.jacfwd(self.B)(x))))

    def field(self, xyz):
        """B (T) and J = curl(B) / mu_0 (A/m^2) at Cartesian points xyz, shape (n, 3).

        J is the curl of the closed form of B, by automatic differentiation.
        """
        B, dB = self._B_and_grad(xyz)  # dB[:, i, j] = dB_i / dx_j
        curl = jnp.stack(
            [
                dB[:, 2, 1] - dB[:, 1, 2],
                dB[:, 0, 2] - dB[:, 2, 0],
                dB[:, 1, 0] - dB[:, 0, 1],
            ],
            axis=-1,
        )
        return np.asarray(B), np.asarray(curl) / mu_0

    def current(self, rho, num_theta=128):
        """Toroidal current (A) enclosed by the surfaces rho, DESC's ``"current"``.

        The circulation of B along theta at phi = 0, divided by mu_0.
        """
        theta = 2 * np.pi * np.arange(num_theta) / num_theta
        R, Z = self.section(np.atleast_1d(rho)[:, None], theta, 0.0)
        ik = 1j * np.fft.fftfreq(num_theta, 1 / num_theta)
        R_t = np.fft.ifft(ik * np.fft.fft(R)).real
        Z_t = np.fft.ifft(ik * np.fft.fft(Z)).real
        B, _ = self.field(np.stack([R, 0 * R, Z], axis=-1).reshape(-1, 3))
        B = B.reshape(*R.shape, 3)
        return 2 * np.pi * np.mean(B[..., 0] * R_t + B[..., 2] * Z_t, axis=-1) / mu_0


class Iota2Equilibrium(_ExactEquilibrium):
    """Exact equilibrium with iota = 2 on every flux surface.

    Every field line closes after one toroidal turn, so theta = 2 phi - alpha, with
    alpha the polar angle of the field line label (u, v) about the magnetic axis, is a
    straight field line angle: in DESC's coordinates iota = 2 and lambda = 0.

    Parameters
    ----------
    eps : float
        Horizontal elongation, 0 < eps < 1.
    delta : float
        Flux label psi of the boundary, which sets its minor radius.
    tau0 : float
        Stellarator symmetry breaking, -1 < tau0 < 1.

    """

    def __init__(self, eps=0.25, delta=1 / 64, tau0=0.0):
        self.eps, self.delta, self.tau0 = eps, delta, tau0
        self.sym = tau0 == 0
        self.a, self.b = np.sqrt(1 + eps), np.sqrt(1 - eps)
        self.omega2 = 1 - tau0**2  # the papers' Omega^2
        # the toroidal flux is pi a b psi / Omega^2, so psi = delta rho^2
        self.Psi = np.pi * self.a * self.b * delta / self.omega2
        p0 = 2 * delta / (self.omega2 * mu_0)
        self.pressure = PowerSeriesProfile([p0, -p0], modes=[0, 2])
        self.iota = PowerSeriesProfile([2.0])

    def B(self, xyz):
        """B (T) at the Cartesian point xyz."""
        a, b, t = self.a, self.b, self.tau0
        x, y, z = xyz
        s = (x / a) ** 2 + (y / b) ** 2
        F = jnp.sqrt(1 - (1 - s) ** 2 - 4 * z**2 - 4 * t * s * z)
        return jnp.stack(
            [
                ((2 * z + t * s) * x - a / b * F * y) / s,
                ((2 * z + t * s) * y + b / a * F * x) / s,
                1 - s - 2 * t * z,
            ]
        )

    def section(self, rho, theta, phi):
        """R, Z (m) of the point rho, theta in the plane phi."""
        o, t = np.sqrt(self.omega2), self.tau0
        alpha = 2 * phi - theta
        u = -self.omega2 * self.eps / 2 + np.sqrt(self.delta) * rho * np.cos(alpha)
        v = np.sqrt(self.delta) * rho * np.sin(alpha)
        ell = np.sqrt((1 + np.sqrt(1 - 4 * (u**2 + v**2))) / 2)
        # the field line (u, v) is x = A cos(s) + C sin(s), y = D cos(s) + E sin(s)
        A, C = self.a * (ell + u / ell) / o, self.a * v / ell / o
        D, E = self.b * v / ell / o, self.b * (ell - u / ell) / o
        # and crosses the half plane phi at the field line parameter s
        cs = E * np.cos(phi) - C * np.sin(phi)
        sn = A * np.sin(phi) - D * np.cos(phi)
        norm = np.hypot(cs, sn)
        cs, sn = cs / norm, sn / norm
        # cos and sin of 2 s + arcsin(tau0)
        c2, s2 = cs**2 - sn**2, 2 * sn * cs
        Z = (-t / 2 + v * (c2 * o - s2 * t) - u * (s2 * o + c2 * t)) / self.omega2
        return np.hypot(A * cs + C * sn, D * cs + E * sn), Z


class ShearedIotaEquilibrium(_ExactEquilibrium):
    """Exact equilibrium with sheared iota.

    Its flux surfaces are the circles X = -k cos(chi), Y = k sin(chi) of the plane of
    X = lam B_z and Y = -sin(lam z), with flux label psi = k^2 / 2. DESC's theta is
    chi, which is not a straight field line angle, so lambda is not zero.

    Parameters
    ----------
    eps, S, lam : float
        Focal distance, axis and vertical wavenumber parameters, all positive.
    k_b : float
        k of the boundary, 0 < k_b < sin(min(S, pi/2)).
    tau1 : float
        Stellarator symmetry breaking.

    """

    def __init__(self, eps=0.6, S=2.2, lam=2.97, k_b=0.2, tau1=0.0):
        self.eps, self.S, self.lam, self.k_b, self.tau1 = eps, S, lam, k_b, tau1
        self.sym = tau1 == 0
        # The toroidal flux and iota against kappa = k^2 are smooth, and so is kappa
        # against DESC's s = rho^2 = flux / Psi.
        n = 33
        kappa = k_b**2 * (1 - np.cos(np.pi * (np.arange(n) + 0.5) / n)) / 2
        dflux, iota = self._rates(kappa)
        flux = Chebyshev.fit(kappa, dflux, n - 1).integ(lbnd=0)
        self.Psi = flux(k_b**2)
        s = flux(kappa) / self.Psi
        coef = lambda f: Polynomial.fit(s, f, 16).convert().coef  # noqa: E731
        # k = rho sqrt(q(rho^2)) with q = kappa / s, so k is 0 on the axis exactly
        self._q = Polynomial(coef(kappa / s))
        # pressure (psi_b - psi) / lam^2, zero at the boundary
        p = coef((k_b**2 - kappa) / (2 * lam**2 * mu_0))
        self.pressure = PowerSeriesProfile(p, modes=2 * np.arange(p.size))
        iota = coef(iota)
        self.iota = PowerSeriesProfile(iota, modes=2 * np.arange(iota.size))

    def _h_nu(self, k, chi, zeta):
        """Confocal h(sigma) and nu at k, chi and the papers' toroidal angle zeta."""
        X, Y = -k * np.cos(chi), k * np.sin(chi)
        nu = self.eps / 2 * np.sin(2 * zeta) + self.tau1
        sigma = (
            self.S
            + np.arctan(np.tanh(nu) * Y / np.sqrt(1 - Y**2))
            - np.arcsin(X / np.sqrt(np.cosh(nu) ** 2 - Y**2))
        )
        return np.sqrt(4 * sigma**2 + self.eps**2), nu

    def _rates(self, kappa, n=64):
        """d(toroidal flux)/d(kappa) and iota on the surfaces kappa = k^2.

        With the coordinate Jacobian sqrt(g) of k, chi, zeta and with B.grad(zeta) and
        B.grad(chi) from Landreman's section 3, B.grad(zeta) sqrt(g) = k / (2 lam h c)
        and B.grad(chi) sqrt(g) = k G / (lam h T), c = sqrt(1 - Y^2), whose surface
        integrals are the rates in k of the toroidal and poloidal fluxes.
        """
        k = np.sqrt(kappa)[:, None, None]
        chi, zeta = np.meshgrid(*2 * [2 * np.pi * np.arange(n) / n], indexing="ij")
        h, nu = self._h_nu(k, chi, zeta)
        c = np.sqrt(1 - (k * np.sin(chi)) ** 2)
        G = (h + self.eps * np.cos(2 * zeta)) / 2
        T = np.sqrt(np.cosh(nu) ** 2 - k**2)
        toroidal = np.mean(1 / (h * c), axis=(1, 2))
        poloidal = np.mean(G / (h * T), axis=(1, 2))
        return np.pi / (2 * self.lam) * toroidal, 2 * poloidal / toroidal

    def B(self, xyz):
        """B (T) at the Cartesian point xyz."""
        x, y, z = xyz
        w, wb = x + 1j * y, x - 1j * y
        K = wb * jnp.sqrt(1 + self.eps / wb**2)
        Xi = w * K + np.pi / 2 - self.S + 1j * self.tau1
        e = jnp.exp(-1j * self.lam * z)
        Bxy = e * 1j * jnp.sin(Xi) / (2 * K)
        return jnp.stack([Bxy.real, Bxy.imag, (e * jnp.cos(Xi)).real / self.lam])

    def section(self, rho, theta, phi):
        """R, Z (m) of the point rho, theta in the plane phi."""
        rho, chi, phi = np.broadcast_arrays(rho, theta, phi)
        k = rho * np.sqrt(self._q(rho**2))
        # x = a cos(zeta), y = b sin(zeta) on the ellipse sigma, by fixed point
        # iteration for zeta, a contraction as sigma depends on zeta at order k eps
        zeta = phi
        for _ in range(20):
            h, _ = self._h_nu(k, chi, zeta)
            a, b = np.sqrt((h - self.eps) / 2), np.sqrt((h + self.eps) / 2)
            zeta = np.arctan2(a * np.sin(phi), b * np.cos(phi))
        R = np.hypot(a * np.cos(zeta), b * np.sin(zeta))
        return R, -np.arcsin(k * np.sin(chi)) / self.lam


def fit_exact_solution(exact, L, M, N):
    """Equilibrium at resolution L, M, N fitted to the exact solution, never solved.

    R and Z are least squares fits to the exact flux surfaces. Lambda follows from the
    exact field on them through B^zeta sqrt(g) / psi_r = 1 + lambda_theta and
    B^theta sqrt(g) / psi_r = iota - lambda_zeta, integrated on each surface.
    """
    eq = Equilibrium(
        L=L,
        M=M,
        N=N,
        NFP=exact.NFP,
        sym=exact.sym,
        Psi=exact.Psi,
        pressure=exact.pressure,
        iota=exact.iota,
    )
    # radial nodes clustered at both ends, none on the axis
    num_rho = 2 * L + 4
    rho = (1 - np.cos(np.pi * (np.arange(num_rho) + 0.5) / num_rho)) / 2
    grid = LinearGrid(rho=rho, M=2 * M + 3, N=2 * N + 3, NFP=exact.NFP)
    R, Z = exact.section(*grid.nodes.T)
    eq.set_initial_guess(grid, R, Z, ensure_nested=False)

    names = ["R", "phi", "Z", "sqrt(g)", "e^theta", "e^zeta", "iota", "psi_r"]
    data = eq.compute(names, grid=grid)
    B, _ = exact.field(rpz2xyz(np.stack([data["R"], data["phi"], data["Z"]], axis=-1)))
    B = xyz2rpz_vec(B, phi=data["phi"])
    lam_t = dot(B, data["e^zeta"]) * data["sqrt(g)"] / data["psi_r"] - 1
    lam_z = data["iota"] - dot(B, data["e^theta"]) * data["sqrt(g)"] / data["psi_r"]
    # least squares lambda with zero mean in theta, zeta on each surface
    shape = (grid.num_theta, grid.num_rho, grid.num_zeta)
    lam_t, lam_z = (
        np.fft.fft2(np.reshape(f, shape, order="F"), axes=(0, 2))
        for f in (lam_t, lam_z)
    )
    m = np.fft.fftfreq(grid.num_theta, 1 / grid.num_theta)[:, None, None]
    n = exact.NFP * np.fft.fftfreq(grid.num_zeta, 1 / grid.num_zeta)
    mn = np.where(m**2 + n**2 > 0, m**2 + n**2, 1)
    lam = np.fft.ifft2(-1j * (m * lam_t + n * lam_z) / mn, axes=(0, 2)).real
    eq.set_initial_guess(grid, R, Z, lam.ravel(order="F"), ensure_nested=False)
    # fitting to points updates neither the boundary nor the axis, which a solve uses
    eq.surface = eq.get_surface_at(rho=1)
    eq.axis = eq.get_axis()
    return eq


def _compare(exact, eq, grid, names):
    """Return DESC's data on grid, and the squared errors and norms of B and J."""
    data = eq.compute(["R", "phi", "Z", "B", "J", "sqrt(g)", *names], grid=grid)
    exact_BJ = exact.field(
        rpz2xyz(np.stack([data["R"], data["phi"], data["Z"]], axis=-1))
    )
    squares = {
        key: (
            np.sum((rpz2xyz_vec(data[key], phi=data["phi"]) - f) ** 2, axis=-1),
            np.sum(f**2, axis=-1),
        )
        for key, f in zip("BJ", exact_BJ)
    }
    return data, squares


def volume_errors(exact, eq):
    """Errors of the quantities DESC computes on eq against the exact solution.

    Returns
    -------
    errors : dict
        ``"F"`` is <|F|>_vol / <|grad(|B|^2)|/2mu0>_vol; ``"B"`` and ``"J"`` volume RMS
        errors relative to the volume RMS values; ``"current"`` the largest relative
        error of DESC's ``"current"`` over 20 surfaces; and ``"axis"`` the largest
        distance (m) of the magnetic axis from the exact one.

    """
    grid = QuadratureGrid(L=2 * eq.L + 4, M=2 * eq.M + 4, N=2 * eq.N + 4, NFP=eq.NFP)
    data, squares = _compare(exact, eq, grid, ["<|F|>_vol", "<|grad(|B|^2)|/2mu0>_vol"])
    w = data["sqrt(g)"] * grid.weights
    errors = {"F": data["<|F|>_vol"] / data["<|grad(|B|^2)|/2mu0>_vol"]}
    for key, (e, norm) in squares.items():
        errors[key] = np.sqrt(np.sum(w * e) / np.sum(w * norm))

    rho = np.linspace(0.05, 1, 20)
    grid = LinearGrid(rho=rho, M=2 * eq.M + 4, N=2 * eq.N + 4, NFP=eq.NFP)
    current = grid.compress(eq.compute("current", grid=grid)["current"])
    errors["current"] = np.max(np.abs(current / exact.current(rho) - 1))

    grid = LinearGrid(rho=0.0, theta=0.0, N=4 * eq.N + 8, NFP=eq.NFP)
    data = eq.compute(["R", "Z"], grid=grid)
    R, Z = exact.axis(grid.nodes[:, 2])
    errors["axis"] = np.max(np.hypot(data["R"] - R, data["Z"] - Z))
    return errors


def surface_errors(exact, eq, rho):
    """Errors on the surfaces rho of the quantities DESC computes on eq.

    Returns
    -------
    errors : dict
        On each surface, ``"F"`` is <|F|> / <|grad(|B|^2)|/2mu0>_vol; ``"B"`` and
        ``"J"`` RMS errors relative to the RMS values; and ``"current"`` the error of
        DESC's ``"current"`` relative to the current.

    """
    grid = LinearGrid(rho=rho, M=3 * eq.M + 2, N=3 * eq.N + 2, NFP=eq.NFP)
    data, squares = _compare(exact, eq, grid, ["|F|", "current"])
    average = lambda f: surface_averages(  # noqa: E731
        grid, f, data["sqrt(g)"], expand_out=False
    )
    F = eq.compute("<|grad(|B|^2)|/2mu0>_vol")["<|grad(|B|^2)|/2mu0>_vol"]
    errors = {"F": average(data["|F|"]) / F}
    for key, (e, norm) in squares.items():
        errors[key] = np.sqrt(average(e) / average(norm))
    current_exact = exact.current(rho)
    errors["current"] = np.abs(grid.compress(data["current"]) / current_exact - 1)
    return errors
