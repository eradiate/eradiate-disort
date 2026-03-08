import numpy as np
from eradiate.experiments import AtmosphereExperiment
from eradiate.units import unit_registry as ureg


def no_atmo(sza: float = 0.0):
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
    return AtmosphereExperiment(
        geometry={
            "type": "plane_parallel",
            "toa_altitude": 1.0 * ureg.km,
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
