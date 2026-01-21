import logging  # noqa: F401

import eradiate
import numpy as np
from eradiate.experiments import AtmosphereExperiment
from eradiate.units import unit_registry as ureg

import eradiate_disort as ed

# logging.basicConfig(level=logging.DEBUG)
eradiate.set_mode("ckd")

exp = AtmosphereExperiment(
    geometry={
        "type": "plane_parallel",
        "zgrid": (np.arange(0, 120.001, 1.0) * ureg.km).to("m"),
    },
    atmosphere={
        "type": "molecular",
    },
    illumination={
        "type": "directional",
        "zenith": 30.0,
        "azimuth": 0.0,
    },
    measures={
        "type": "mdistant",
        "construct": "hplane",
        "azimuth": 0.0,
        "zeniths": np.arange(-75.0, 76.0, 5.0),
        "srf": {"type": "delta", "wavelengths": [550.0]},
        # "srf": {"type": "uniform", "wmin": 525.0, "wmax": 575.0},
    },
)

backend = ed.EradiateDisortBackend(verbose=True)
backend.process(exp)
