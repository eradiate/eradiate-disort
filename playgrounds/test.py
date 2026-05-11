import eradiate
import numpy as np
from eradiate.experiments import AtmosphereExperiment
from eradiate.units import unit_registry as ureg

import eradiate_disort as ed

eradiate.set_mode("ckd")

# Load surface spectrum
from eradiate.scenes.spectra import InterpolatedSpectrum

albedo_data = np.loadtxt(
    eradiate.fresolver.resolve("spectra/HAMSTER_spectral_albedo_Gobabeb_015.txt"),
    skiprows=1,
)
albedo_spectrum = InterpolatedSpectrum(
    wavelengths=albedo_data[:, 0], values=albedo_data[:, 1]
)


def experiment(backend, wmin=600.0, wmax=610.0, tau_ref=0.0):
    srf = {"type": "uniform", "wmin": wmin, "wmax": wmax}
    angles = {"construct": "hplane", "azimuth": 75.0, "zeniths": [60.0]}
    measure = (
        {"type": "mdistant", **angles, "srf": srf}
        if backend == "mitsuba"
        else {"type": "disort", **angles, "srf": srf}
    )

    particle_layer = (
        {
            "tau_ref": tau_ref,
            "w_ref": 550.0,
            "bottom": 0.0,
            "top": 3.0 * ureg.km,
            "distribution": "uniform",
            "dataset": "soot.mie-aer_core_v2",
        }
        if tau_ref
        else []
    )

    exp = AtmosphereExperiment(
        geometry={
            "type": "plane_parallel",
            "toa_altitude": 120.0 * ureg.km,
            "zgrid": np.linspace(0, 120, 121) * ureg.km,
        },
        surface={"type": "lambertian", "reflectance": albedo_spectrum},
        atmosphere={
            "type": "heterogeneous",
            "molecular_atmosphere": {"absorption_data": "panellus"},
            "particle_layers": particle_layer,
        },
        illumination={
            "type": "directional",
            "zenith": 30.0,
            "azimuth": 160.0,
        },
        measures=measure,
    )
    exp.integrator.moment = True

    return exp


exp_kwargs = {"wmin": 650.0, "wmax": 660.0, "tau_ref": 0.1}
exp_disort = experiment("disort", **exp_kwargs)
backend = ed.EradiateDisortBackend()
result_disort = backend.run(exp_disort)
