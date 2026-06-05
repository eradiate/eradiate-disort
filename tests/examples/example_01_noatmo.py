# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags
#     formats: tests/examples//py:percent,docs/examples//ipynb
#     notebook_metadata_filter: kernelspec
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: eradiate-disort (pixi)
#     language: python
#     name: eradiate-disort
# ---

# %% [markdown]
# # Reflective surface testing
#
# This notebook tests the backend setup for a diffuse surface without an
# atmosphere.

# %% tags=["remove-cell"]
# Documentation-specific setup, hidden from notebook output

# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'

import seaborn as sns

sns.set_theme(style="ticks")

# %%
import numpy as np

import eradiate
import matplotlib.pyplot as plt
from eradiate.experiments import AtmosphereExperiment
from eradiate.units import unit_registry as ureg

import eradiate_disort as ed
from eradiate_disort.testing import TestMode
from eradiate_disort.util import disort_reshape_pplane

eradiate.set_mode("ckd")

SPP = 1_000

# %% tags=["remove-cell"]
# Dev-specific setup, hidden from notebook output

plt = TestMode.plt()
_base_spp = TestMode.spp(tutorial=1_000, test=1_000)
SPP = _base_spp // 16 if eradiate.get_mode().is_ckd else _base_spp

# %%
# Experiment parameters
SZA = 30.0
zeniths = np.arange(-75.0, 76.0, 1.0)
srf = {"type": "delta", "wavelengths": [550.0]}
geometry = {
    "type": "plane_parallel",
    "zgrid": ([0.0, 1.0] * ureg.km).to("m"),
    "toa_altitude": 1.0 * ureg.km,
}
surface = {"type": "lambertian", "reflectance": 0.5}
illumination = {"type": "directional", "zenith": SZA, "azimuth": 0.0}

# %%
result = {}

exp = AtmosphereExperiment(
    geometry=geometry,
    surface=surface,
    atmosphere=None,
    illumination=illumination,
    measures={
        "type": "mdistant",
        "id": "toa_mitsuba",
        "construct": "hplane",
        "azimuth": 0.0,
        "zeniths": zeniths,
        "srf": srf,
    },
)
result["mitsuba"] = eradiate.run(exp, spp=SPP)["radiance"].squeeze()

exp = AtmosphereExperiment(
    geometry=geometry,
    surface=surface,
    atmosphere=None,
    illumination=illumination,
    measures={
        "type": "disort",
        "id": "toa_disort",
        "construct": "hplane",
        "azimuth": 0.0,
        "zeniths": zeniths,
        "srf": srf,
    },
)
backend = ed.DisortBackend()
result["disort"] = disort_reshape_pplane(backend.run(exp).sel(z=1.0))

# %%
fig, ax = plt.subplots(1, 1, figsize=(4, 3), layout="constrained")

ax.plot(result["mitsuba"]["vza"], result["mitsuba"], label="Mitsuba")
ax.plot(result["disort"]["vza"], result["disort"], label="CDISORT", ls="--")
ax.set_xlabel("θ [°]")
ax.set_ylabel("Radiance [W/m²/sr]")
ax.set_ylim([-0.05, 0.65])
ax.legend()

plt.show()
