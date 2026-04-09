import eradiate
import numpy as np
import seaborn as sns
from eradiate.experiments import AtmosphereExperiment
from eradiate.units import unit_registry as ureg

import eradiate_disort as ed

eradiate.fresolver.prepend("data")
eradiate.set_mode("ckd")
sns.set_theme(style="ticks")

SPP = 100_000
if eradiate.get_mode().is_ckd:
    SPP //= 16


def full_atmo(
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
        measures={
            "type": "mdistant",
            "construct": "hplane",
            "azimuth": 0.0,
            "zeniths": np.arange(-75.0, 76.0, 1.0),
            "srf": {"type": "delta", "wavelengths": [550.0]},
        },
    )


exp = full_atmo(has_absorption=False, has_scattering=True, surface_reflectance=0.0)
exp.init()
backend = ed.EradiateDisortBackend()
raw = backend.run(exp)
print(raw)
print("has NaN:", bool(raw.isnull().any()))
