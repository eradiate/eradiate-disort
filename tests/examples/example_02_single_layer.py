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
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: eradiate-disort (pixi dev)
#     language: python
#     name: eradiate-disort
# ---

# %% [markdown]
# # Single layer testing
#
# This notebook tests the backend setup for a single layer.

# %% tags=["remove-cell"]
import eradiate
import seaborn as sns

import eradiate_disort as ed
from eradiate_disort.testing import TestMode, cases
from eradiate_disort.testing.util import Result, disort_reshape_pplane

plt = TestMode.plt()

sns.set_theme(style="ticks")
eradiate.set_mode("ckd")

_base_spp = TestMode.spp(tutorial=1_000, test=10_000)
SPP = _base_spp // 16 if eradiate.get_mode().is_ckd else _base_spp
PHASES = ["isotropic", "rayleigh"]

# %%
results = {}

for phase in PHASES:
    result = Result()

    exp = cases.single_layer(30.0, phase, backend="mitsuba")
    result.mitsuba = eradiate.run(exp, spp=SPP)["radiance"].squeeze()

    exp = cases.single_layer(30.0, phase, backend="disort")
    backend = ed.EradiateDisortBackend()
    result.disort = disort_reshape_pplane(backend.run(exp).sel(z=1000.0))

    results[phase] = result


# %%
def plot(results):
    fig, axs = plt.subplots(1, len(results), figsize=(8, 3.5), layout="constrained")

    for k, phase in enumerate(results.keys()):
        ax = axs[k]
        result = results[phase]

        ax.plot(
            result.mitsuba["vza"],
            result.mitsuba,
            label="Mitsuba" if k == 0 else None,
        )
        ax.plot(
            result.disort["vza"],
            result.disort,
            label="CDISORT" if k == 0 else None,
            ls="--",
        )

        ax.set_xlabel("θ [°]")
        ax.set_ylabel("Radiance [W/m²/sr]" if k == 0 else None)
        ax.set_title(phase.title())

    fig.legend(title="Backend", loc="outside upper center", ncol=2)

    return fig, axs


plot(results)
plt.show()
plt.close()

# %%
import attrs
from eradiate.xarray import unstack_mdistant_grid

results = {}

for phase in ["rayleigh"]:
    result = Result()

    exp = attrs.evolve(
        cases.single_layer(30.0, phase, backend="mitsuba"),
        measures=cases._grid_measure(backend="mitsuba"),
    )
    result.mitsuba = unstack_mdistant_grid(
        eradiate.run(exp, spp=SPP)["radiance"]
    ).squeeze()

    exp = attrs.evolve(
        cases.single_layer(30.0, phase, backend="disort"),
        measures=cases._grid_measure(backend="disort"),
    )
    backend = ed.EradiateDisortBackend()
    result.disort = backend.run(exp)["measure/uu"].sel(z=1000.0).squeeze().sortby("vza")

    results[phase] = result


# %%
def plot(results):
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained", sharey=True)

    ax = axs[0]
    results["rayleigh"].mitsuba.plot.imshow(ax=ax, cbar_kwargs={"label": None})
    ax.set_title("Mitsuba")

    ax = axs[1]
    results["rayleigh"].disort.plot.imshow(ax=ax, cbar_kwargs={"label": None})
    ax.set_title("CDISORT")
    ax.set_ylabel(None)

    ax = axs[2]
    (
        (result.disort.assign_coords({k: result.mitsuba[k] for k in ["vza", "vaa"]}))
        - result.mitsuba
    ).plot.imshow(ax=ax)
    ax.set_title("Difference")
    ax.set_ylabel(None)

    return fig, axs


plot(results)
plt.show()
plt.close()
