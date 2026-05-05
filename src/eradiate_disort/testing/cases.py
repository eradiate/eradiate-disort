# SPDX-FileCopyrightText: 2026 Rayference
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import xarray as xr
from eradiate.experiments import AtmosphereExperiment
from eradiate.radprops import ArrayRadProfile, ZGrid
from eradiate.units import unit_registry as ureg

_ZENITHS = np.arange(-75.0, 76.0, 1.0)
_SRF = {"type": "delta", "wavelengths": [550.0]}


def _hplane_measure(backend: str, srf: Optional[dict] = None) -> dict:
    """Return a hemisphere-plane measure dict for the given backend."""
    mtype = "mdistant" if backend == "mitsuba" else "disort"
    result = {
        "type": mtype,
        "construct": "hplane",
        "azimuth": 0.0,
        "zeniths": _ZENITHS,
    }
    if srf is not None:
        result["srf"] = srf
    return result


def _grid_measure(backend: str, srf: Optional[dict] = None) -> dict:
    mtype = "mdistant" if backend == "mitsuba" else "disort"
    result = {
        "type": mtype,
        "construct": "grid",
        "azimuths": np.arange(0, 360, 10),
        "zeniths": np.arange(0, 75.1, 5),
    }
    if srf is not None:
        result["srf"] = srf
    return result


def no_atmo(sza: float = 0.0, backend: Literal["mitsuba", "disort"] = "mitsuba"):
    """
    Generate an experiment for the "No atmosphere" test case series.
    """
    zeniths = np.arange(-75.0, 76.0, 1.0)
    srf = {"type": "delta", "wavelengths": [550.0]}
    if backend == "mitsuba":
        measures = {
            "type": "mdistant",
            "id": "toa_mitsuba",
            "construct": "hplane",
            "azimuth": 0.0,
            "zeniths": zeniths,
            "srf": srf,
        }
    elif backend == "disort":
        measures = {
            "type": "disort",
            "id": "toa_disort",
            "construct": "hplane",
            "azimuth": 0.0,
            "zeniths": zeniths,
            "srf": srf,
        }
    else:
        raise NotImplementedError

    return AtmosphereExperiment(
        geometry={
            "type": "plane_parallel",
            "zgrid": ([0.0, 1.0] * ureg.km).to("m"),
            "toa_altitude": 1.0 * ureg.km,
        },
        surface={"type": "lambertian", "reflectance": 0.5},
        atmosphere=None,
        illumination={"type": "directional", "zenith": sza, "azimuth": 0.0},
        measures=measures,
    )


def single_layer(
    sza: float = 0.0,
    phase: str = "isotropic",
    backend: Literal["mitsuba", "disort"] = "disort",
):
    """
    Generate an experiment for the "Single layer" test case series.
    """
    return AtmosphereExperiment(
        geometry={
            "type": "plane_parallel",
            "toa_altitude": 1.0 * ureg.km,
            "zgrid": np.linspace(0, 1, 2) * ureg.km,
        },
        surface={"type": "lambertian", "reflectance": 0.0},
        atmosphere={"type": "homogeneous", "phase": {"type": phase}},
        illumination={"type": "directional", "zenith": sza, "azimuth": 0.0},
        measures=_hplane_measure(backend, srf=_SRF),
    )


def two_layers(
    sza: float = 30.0,
    has_scattering: bool = True,
    has_absorption: bool = True,
    surface_reflectance: float = 0.0,
    backend: Literal["mitsuba", "disort"] = "mitsuba",
):
    """
    Generate an experiment for the "Two layers" test case series.
    """
    zgrid = ZGrid(np.linspace(0, 100e3, 11) * ureg.m)
    tau_a = 0.5
    tau_s = 0.25
    sigma_a = tau_a / zgrid.total_height
    sigma_s = tau_s / zgrid.total_height  # noqa: F841

    # def sigma_constant(value, zgrid):
    #     z = zgrid.levels
    #     return xr.DataArray(
    #         np.full_like(z.m_as("m"), value.m_as("m^-1")).reshape((1, -1)),
    #         dims=("w", "z"),
    #         coords={
    #             "w": ("w", [550.0], {"units": "nm"}),
    #             "z": ("z", z.m_as("m"), {"units": "m"}),
    #         },
    #         attrs={"units": "1/m"},
    #     )

    def sigma_step(value_a, value_b, h, zgrid):
        z = zgrid.levels
        mask = ~(z < h)
        values = np.zeros_like(z)
        values[mask] = value_a.m_as("m^-1")
        values[~mask] = value_b.m_as("m^-1")
        return xr.DataArray(
            np.full_like(z.m_as("m"), values).reshape((1, -1)),
            dims=("w", "z"),
            coords={
                "w": ("w", [550.0], {"units": "nm"}),
                "z": ("z", z.m_as("m"), {"units": "m"}),
            },
            attrs={"units": "1/m"},
        )

    radprofile = ArrayRadProfile(
        sigma_a=sigma_step(sigma_a, sigma_a / 2, 50 * ureg.km, zgrid),
        sigma_s=sigma_step(sigma_a / 2, sigma_a, 50 * ureg.km, zgrid),
        has_absorption=has_absorption,
        has_scattering=has_scattering,
    )

    return AtmosphereExperiment(
        geometry={
            "type": "plane_parallel",
            "toa_altitude": 100.0 * ureg.km,
            "zgrid": zgrid,
        },
        surface={"type": "lambertian", "reflectance": surface_reflectance},
        atmosphere={
            "type": "molecular",
            "has_scattering": has_scattering,
            "has_absorption": has_absorption,
            "thermoprops": None,
            "radprops_profile": radprofile,
        },
        illumination={"type": "directional", "zenith": sza, "azimuth": 0.0},
        measures=_hplane_measure(backend),
    )


def molecular(
    sza: float = 30.0,
    has_scattering: bool = True,
    has_absorption: bool = True,
    surface_reflectance: float = 0.0,
    backend: Literal["mitsuba", "disort"] = "mitsuba",
):
    """
    Generate an experiment for the "Molecular atmosphere" test case series.
    """
    return AtmosphereExperiment(
        geometry={
            "type": "plane_parallel",
            "toa_altitude": 100.0 * ureg.km,
            "zgrid": np.linspace(0, 100, 101) * ureg.km,
        },
        surface={"type": "lambertian", "reflectance": surface_reflectance},
        atmosphere={
            "type": "molecular",
            "has_scattering": has_scattering,
            "has_absorption": has_absorption,
        },
        illumination={"type": "directional", "zenith": sza, "azimuth": 0.0},
        measures=_hplane_measure(backend),
    )


def aerosols(
    sza: float = 30.0,
    has_scattering: bool = True,
    has_absorption: bool = True,
    surface_reflectance: float = 0.0,
    backend: Literal["mitsuba", "disort"] = "mitsuba",
):
    """
    Generate an experiment for the "Aerosols" test case series.
    """
    return AtmosphereExperiment(
        geometry={
            "type": "plane_parallel",
            "toa_altitude": 100.0 * ureg.km,
            "zgrid": np.linspace(0, 100, 101) * ureg.km,
        },
        surface={"type": "lambertian", "reflectance": surface_reflectance},
        atmosphere={
            "type": "particle_layer",
            "has_scattering": has_scattering,
            "has_absorption": has_absorption,
            "tau_ref": 0.5,
            "particle_properties": "soot.mie-aer_core_v2",
        },
        illumination={"type": "directional", "zenith": sza, "azimuth": 0.0},
        measures=_hplane_measure(backend),
    )


def full_atmo(
    sza: float = 30.0,
    has_scattering: bool = True,
    has_absorption: bool = True,
    surface_reflectance: float = 0.0,
    backend: Literal["mitsuba", "disort"] = "mitsuba",
):
    """
    Generate an experiment for the "Full atmosphere" test case series.
    """
    return AtmosphereExperiment(
        geometry={
            "type": "plane_parallel",
            "toa_altitude": 100.0 * ureg.km,
            "zgrid": np.linspace(0, 100, 101) * ureg.km,
        },
        surface={"type": "lambertian", "reflectance": surface_reflectance},
        atmosphere={
            "type": "heterogeneous",
            "molecular_atmosphere": {
                "has_scattering": has_scattering,
                "has_absorption": has_absorption,
            },
            "particle_layers": {
                "has_scattering": has_scattering,
                "has_absorption": has_absorption,
                "tau_ref": 0.2,
                "particle_properties": "soot.mie-aer_core_v2",
            },
        },
        illumination={"type": "directional", "zenith": sza, "azimuth": 0.0},
        measures=_hplane_measure(backend),
    )
