# SPDX-FileCopyrightText: 2026 Rayference
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Profiling workload for the DISORT backend.

Two scenarios, derived from ``tests/examples/example_06_full_atmo.py`` and
``tests/examples/example_07_spectrum.py`` with the Monte Carlo (Mitsuba) runs
removed:

``angular``
    Molecular + aerosol atmosphere, 100 layers, principal plane with 151
    zeniths, a single default spectral band. Stresses the per-solve angular
    work.

``spectral``
    Molecular atmosphere (CKD absorption database) + optional cloud layer,
    120 layers, a single viewing direction over a wavelength interval.
    Stresses the spectral loop.

Run directly for stage timings, or under a sampling profiler, e.g.::

    py-spy record --native --rate 250 -f raw -o profile.txt -- \
        python scripts/profile_disort.py spectral
"""

from __future__ import annotations

import argparse
import time
from contextlib import contextmanager
from pathlib import Path

import eradiate
import numpy as np
from eradiate.experiments import AtmosphereExperiment
from eradiate.scenes.spectra import InterpolatedSpectrum
from eradiate.units import unit_registry as ureg

import eradiate_disort as ed

REPO_ROOT = Path(__file__).parent.parent
TIMINGS: dict[str, float] = {}


@contextmanager
def timer(label: str):
    t0 = time.perf_counter()
    yield
    TIMINGS[label] = TIMINGS.get(label, 0.0) + time.perf_counter() - t0


def make_angular_experiment() -> AtmosphereExperiment:
    return AtmosphereExperiment(
        geometry={
            "type": "plane_parallel",
            "toa_altitude": 100.0 * ureg.km,
            "zgrid": np.linspace(0, 100, 101) * ureg.km,
        },
        surface={"type": "lambertian", "reflectance": 0.0},
        atmosphere={
            "type": "heterogeneous",
            "molecular_atmosphere": {"has_scattering": True, "has_absorption": True},
            "particle_layers": {
                "has_scattering": True,
                "has_absorption": True,
                "tau_ref": 0.2,
                "particle_properties": "govaerts_2021-continental",
            },
        },
        illumination={"type": "directional", "zenith": 30.0, "azimuth": 0.0},
        measures={
            "type": "disort",
            "construct": "hplane",
            "azimuth": 0.0,
            "zeniths": np.arange(-75.0, 76.0, 1.0),
        },
    )


def make_spectral_experiment(
    wmin: float, wmax: float, absorption_db: str, tau_ref: float
) -> AtmosphereExperiment:
    albedo_data = np.loadtxt(
        eradiate.fresolver.resolve("spectra/HAMSTER_spectral_albedo_Gobabeb_015.txt"),
        skiprows=1,
    )
    albedo_spectrum = InterpolatedSpectrum(
        wavelengths=albedo_data[:, 0], values=albedo_data[:, 1]
    )

    particle_layers = (
        {
            "tau_ref": tau_ref,
            "w_ref": 550.0,
            "bottom": 0.0,
            "top": 3.0 * ureg.km,
            "distribution": "uniform",
            "dataset": "wc.sol.mie_reff.010-aer_core_v2",
        }
        if tau_ref
        else []
    )

    return AtmosphereExperiment(
        geometry={
            "type": "plane_parallel",
            "toa_altitude": 120.0 * ureg.km,
            "zgrid": np.linspace(0, 120, 121) * ureg.km,
        },
        surface={"type": "lambertian", "reflectance": albedo_spectrum},
        atmosphere={
            "type": "heterogeneous",
            "molecular_atmosphere": {"absorption_data": absorption_db},
            "particle_layers": particle_layers,
        },
        illumination={"type": "directional", "zenith": 30.0, "azimuth": 160.0},
        measures={
            "id": "disort",
            "type": "disort",
            "construct": "hplane",
            "azimuth": 75.0,
            "zeniths": [60.0],
            "srf": {"type": "uniform", "wmin": wmin, "wmax": wmax},
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=["angular", "spectral"])
    parser.add_argument("--nstr", type=int, default=16)
    parser.add_argument("--nmom", type=int, default=16)
    parser.add_argument(
        "--intensity-correction",
        default="buras_emde",
        choices=["buras_emde", "nakajima_tanaka"],
    )
    parser.add_argument("--wmin", type=float, default=600.0)
    parser.add_argument("--wmax", type=float, default=650.0)
    parser.add_argument("--absorption-db", default="monotropa")
    parser.add_argument("--tau-ref", type=float, default=0.0)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    with timer("import + data setup"):
        eradiate.fresolver.prepend(REPO_ROOT / "tests" / "data")
        eradiate.set_mode("ckd")

    with timer("scene construction"):
        if args.scenario == "angular":
            exp = make_angular_experiment()
        else:
            exp = make_spectral_experiment(
                args.wmin, args.wmax, args.absorption_db, args.tau_ref
            )

    for _ in range(args.repeat):
        backend = ed.DisortBackend(
            nstr=args.nstr,
            nmom=args.nmom,
            intensity_correction=args.intensity_correction,
        )
        with timer("validate"):
            backend.validate(exp)
        with timer("process"):
            backend.process(exp)
        with timer("postprocess"):
            result = backend.postprocess(exp)

    n_ctx = len(backend._results)
    print()
    print(
        f"scenario: {args.scenario}, spectral iterations: {n_ctx}, "
        f"repeat: {args.repeat}"
    )
    total = sum(TIMINGS.values())
    for label, dt in TIMINGS.items():
        print(f"  {label:22s} {dt:8.3f} s  ({100 * dt / total:5.1f} %)")
    print(f"  {'TOTAL':22s} {total:8.3f} s")
    per_iter = TIMINGS["process"] / (n_ctx * args.repeat) * 1e3
    print(f"  per spectral iteration: {per_iter:.2f} ms")
    print(result)


if __name__ == "__main__":
    main()
