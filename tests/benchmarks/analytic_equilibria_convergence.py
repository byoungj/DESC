"""Convergence of DESC on exact 3D MHD equilibria, without solving.

    python -m tests.benchmarks.analytic_equilibria_convergence --out DIR
                                                              [--res 4 5 ... 16]

At each L = M = N in --res, ``fit_exact_solution`` fits each stellarator symmetric
family of tests/analytic_equilibria.py on DESC's basis, and DESC's force balance,
field, current density, enclosed current and magnetic axis are compared with the
closed forms. Two figures go to DIR, which should be outside the repository:

- analytic_equilibria_rho.png: errors on each surface against rho at every third
  resolution, one row per family;
- analytic_equilibria_convergence.png: errors over the volume against resolution.

The errors are the truncation errors of DESC's basis, so they should fall spectrally
with resolution, with B ahead of its derivatives J and F, and stay as small toward
the magnetic axis as in the bulk.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..analytic_equilibria import (
    Iota2Equilibrium,
    ShearedIotaEquilibrium,
    fit_exact_solution,
    surface_errors,
    volume_errors,
)

FAMILIES = {"iota = 2": Iota2Equilibrium(), "sheared iota": ShearedIotaEquilibrium()}
LABELS = {
    "F": r"$\langle|\mathbf{J}\times\mathbf{B}-\nabla p|\rangle"
    r"\,/\,\langle|\nabla B^2/2\mu_0|\rangle_V$",
    "B": r"$\|\mathbf{B}-\mathbf{B}_\mathrm{exact}\|/\|\mathbf{B}_\mathrm{exact}\|$",
    "J": r"$\|\mathbf{J}-\mathbf{J}_\mathrm{exact}\|/\|\mathbf{J}_\mathrm{exact}\|$",
    "current": r"$|I-I_\mathrm{exact}|\,/\,|I_\mathrm{exact}|$",
    "axis": r"axis $\max|\mathbf{r}-\mathbf{r}_\mathrm{exact}|$ (m)",
}
RHO = np.geomspace(5e-3, 1, 32)


def main():
    """Run the scan and plot it."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--res", type=int, nargs="+", default=list(range(4, 17)))
    parser.add_argument("--out", type=Path, required=True, help="directory for figures")
    args = parser.parse_args()

    volume = {name: {} for name in FAMILIES}
    surface = {name: {} for name in FAMILIES}
    for res in args.res:
        for name, exact in FAMILIES.items():
            eq = fit_exact_solution(exact, res, res, res)
            volume[name][res] = volume_errors(exact, eq)
            if res in args.res[::3]:
                surface[name][res] = surface_errors(exact, eq, RHO)
            print(
                f"{name} L = M = N = {res}: "
                + ", ".join(f"{k} {v:.2e}" for k, v in volume[name][res].items()),
                flush=True,
            )
    args.out.mkdir(parents=True, exist_ok=True)

    keys = list(LABELS)[:-1]
    fig, axs = plt.subplots(
        len(FAMILIES), len(keys), figsize=(14, 6), sharex=True, constrained_layout=True
    )
    for row, name in zip(axs, FAMILIES):
        for ax, key in zip(row, keys):
            for res, errors in surface[name].items():
                ax.loglog(RHO, errors[key], label=f"L = M = N = {res}")
            ax.set_title(f"{name}: {LABELS[key]}", fontsize=9)
            ax.grid(True, which="major", alpha=0.3)
    for ax in axs[-1]:
        ax.set_xlabel(r"$\rho$")
    axs[0, 0].legend(fontsize=8)
    fig.savefig(args.out / "analytic_equilibria_rho.png", dpi=150)

    fig, axs = plt.subplots(1, len(LABELS), figsize=(16, 3.5), constrained_layout=True)
    for ax, key in zip(axs, LABELS):
        for name in FAMILIES:
            res = list(volume[name])
            ax.semilogy(res, [volume[name][r][key] for r in res], "o-", label=name)
        ax.set_title(LABELS[key], fontsize=9)
        ax.set_xticks(args.res[::3])
        ax.set_xlabel("L = M = N")
        ax.grid(True, which="major", alpha=0.3)
    axs[0].legend(fontsize=8)
    fig.savefig(args.out / "analytic_equilibria_convergence.png", dpi=150)


if __name__ == "__main__":
    main()
