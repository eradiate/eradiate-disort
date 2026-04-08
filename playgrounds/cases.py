import numpy as np
import xarray as xr
from eradiate.experiments import AtmosphereExperiment
from eradiate.radprops import ArrayRadProfile, ZGrid
from eradiate.units import unit_registry as ureg


def no_atmo(sza: float = 0.0):
    """
    Generate an experiment for the "No atmosphere" test case series.
    """
    return AtmosphereExperiment(
        geometry={
            "type": "plane_parallel",
            "zgrid": ([0.0, 1.0] * ureg.km).to("m"),
            "toa_altitude": 1.0 * ureg.km,
        },
        surface={"type": "lambertian", "reflectance": 0.5},
        atmosphere=None,
        illumination={"type": "directional", "zenith": sza, "azimuth": 0.0},
        measures={
            "type": "mdistant",
            "construct": "hplane",
            "azimuth": 0.0,
            "zeniths": np.arange(-75.0, 76.0, 1.0),
            "srf": {"type": "delta", "wavelengths": [550.0]},
        },
    )


def single_layer(sza: float = 0.0, phase: str = "isotropic"):
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
        measures={
            "type": "mdistant",
            "construct": "hplane",
            "azimuth": 0.0,
            "zeniths": np.arange(-75.0, 76.0, 1.0),
            "srf": {"type": "delta", "wavelengths": [550.0]},
        },
    )


def two_layers(
    sza: float = 30.0,
    has_scattering: bool = True,
    has_absorption: bool = True,
    surface_reflectance: float = 0.0,
):
    """
    Generate an experiment for the "Two layers" test case series.
    """
    zgrid = ZGrid(np.linspace(0, 100e3, 11) * ureg.m)
    tau_a = 0.5
    tau_s = 0.25
    sigma_a = tau_a / zgrid.total_height
    sigma_s = tau_s / zgrid.total_height

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
        measures={
            "type": "mdistant",
            "construct": "hplane",
            "azimuth": 0.0,
            "zeniths": np.arange(-75.0, 76.0, 1.0),
            "srf": {"type": "delta", "wavelengths": [550.0]},
        },
    )


def molecular(
    sza: float = 30.0,
    has_scattering: bool = True,
    has_absorption: bool = True,
    surface_reflectance: float = 0.0,
):
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
        measures={
            "type": "mdistant",
            "construct": "hplane",
            "azimuth": 0.0,
            "zeniths": np.arange(-75.0, 76.0, 1.0),
            "srf": {"type": "delta", "wavelengths": [550.0]},
        },
    )


def aerosols(
    sza: float = 30.0,
    has_scattering: bool = True,
    has_absorption: bool = True,
    surface_reflectance: float = 0.0,
):
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
        measures={
            "type": "mdistant",
            "construct": "hplane",
            "azimuth": 0.0,
            "zeniths": np.arange(-75.0, 76.0, 1.0),
            "srf": {"type": "delta", "wavelengths": [550.0]},
        },
    )
