# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags
#     formats: py:percent
#     notebook_metadata_filter: kernelspec
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: eradiate-disort (pixi)
#     language: python
#     name: eradiate-disort
# ---

# %% [markdown]
# # Full spectrum example
#
# This notebook compares the output of Eradiate's Mitsuba and DISORT backends for spectral computations.

# %%
import eradiate
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from eradiate.experiments import AtmosphereExperiment
from eradiate.units import unit_registry as ureg

import eradiate_disort as ed

eradiate.set_mode("ckd")
sns.set_theme(style="ticks")


# %%
def experiment(backend, wmin=600.0, wmax=650.0):
    srf = {"type": "uniform", "wmin": wmin, "wmax": wmax}
    angles = {"construct": "hplane", "azimuth": 75.0, "zeniths": [60.0]}
    measure = (
        {"type": "mdistant", **angles, "srf": srf}
        if backend == "mitsuba"
        else {"type": "disort", **angles, "srf": srf}
    )

    exp = AtmosphereExperiment(
        geometry={
            "type": "plane_parallel",
            "toa_altitude": 120.0 * ureg.km,
            "zgrid": np.linspace(0, 120, 121) * ureg.km,
        },
        atmosphere={
            "type": "heterogeneous",
            "molecular_atmosphere": {"absorption_data": "panellus"},
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


# %%
exp_mitsuba = experiment("mitsuba")
result_mitsuba = eradiate.run(exp_mitsuba, spp=10_000)

# %%
exp_disort = experiment("disort")
backend = ed.EradiateDisortBackend()
result_disort = backend.run(exp_disort)

# %%
# Extract irradiance
irradiance = result_mitsuba["irradiance"].squeeze()

# Extract Mitsuba radiance
radiance_mitsuba = result_mitsuba["radiance"].squeeze()
radiance_mitsuba_std = np.sqrt(result_mitsuba["radiance_var"].squeeze())

# Compute Mitsuba BRF standard deviation
brf_mitsuba = result_mitsuba["brf"].squeeze()
x = np.pi / result_mitsuba["irradiance"]
brf_mitsuba_std = np.sqrt((result_mitsuba["radiance_var"] * x**2).squeeze())

# Compute DISORT BRF
radiance_disort = result_disort["measure"]["uu"].isel(z=0).squeeze()
irradiance_disort = result_disort["measure"]["rfldir"].isel(z=0).squeeze()
cos_theta_s = np.cos(exp_disort.illumination.zenith.m_as("rad"))
brf_disort = radiance_disort / irradiance_disort * np.pi

# %%
fig, axs = plt.subplots(2, 1, height_ratios=[0.5, 1], sharex=True, layout="constrained")

ax = axs[0]
x = irradiance["w"].values
y = irradiance.values
ax.step(x, y, where="mid", label="Solar horizontal irradiance", c="C2")

ax.set_ylabel("Irradiance\n[W/m²/nm]")
ax.legend()

ax = axs[1]
x = radiance_mitsuba["w"].values
y = radiance_mitsuba.values
y_std = radiance_mitsuba_std.values
ymin = y - y_std * 2.0
ymax = y + y_std * 2.0

ax.step(x, y, where="mid", label="Mitsuba")
ax.fill_between(x, ymin, ymax, step="mid", alpha=0.25)

x = radiance_disort["w"].values
y = radiance_disort.values
ax.step(x, y, where="mid", label="DISORT")

ax.set_xlabel("Wavelength [nm]")
ax.set_ylabel("TOA radiance\n[W/m²/sr/nm]")
ax.legend()

plt.show()
plt.close()

# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 3))

x = brf_mitsuba["w"].values
y = brf_mitsuba.values
y_std = brf_mitsuba_std.values
ymin = y - y_std * 2.0
ymax = y + y_std * 2.0

ax.step(x, y, where="mid", label="Mitsuba")
ax.fill_between(x, ymin, ymax, step="mid", alpha=0.25)

x = brf_disort["w"].values
y = brf_disort.values
ax.step(x, y, where="mid", label="DISORT")

ax.set_xlabel("Wavelength [nm]")
ax.set_ylabel("TOA BRF [—]")
ax.legend()

plt.show()
plt.close()
