# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags
#     formats: tests/notebooks//py:percent,docs/examples//ipynb
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
# # Full atmosphere testing
#
# This notebook tests the backend setup for an atmosphere with molecular and aerosol
# components.

# %% tags=["remove-cell"]
import eradiate
import matplotlib.pyplot as plt
import seaborn as sns
from eradiate.contexts import KernelContext

import eradiate_disort as ed
from eradiate_disort.testing import TestMode
from eradiate_disort.testing.cases import full_atmo
from eradiate_disort.testing.util import Result, disort_reshape_pplane

eradiate.fresolver.prepend("../data")
eradiate.set_mode("ckd")
sns.set_theme(style="ticks")

_base_spp = TestMode.spp(tutorial=1_000, test=10_000)
SPP = _base_spp // 16 if eradiate.get_mode().is_ckd else _base_spp

exp = full_atmo(sza=30.0)
ctx = KernelContext()
exp.atmosphere.eval_radprops(ctx.si, optional_fields=True)

# %%
results = {}
CASES = {
    "scattering": {
        "has_absorption": False,
        "has_scattering": True,
        "surface_reflectance": 0.0,
    },
    "absorption": {
        "has_absorption": True,
        "has_scattering": False,
        "surface_reflectance": 1.0,
    },
    "full_black": {
        "has_absorption": True,
        "has_scattering": True,
        "surface_reflectance": 0.0,
    },
    "full_white": {
        "has_absorption": True,
        "has_scattering": True,
        "surface_reflectance": 1.0,
    },
}

for case_id, kwargs in CASES.items():
    print(f"Processing case {case_id!r}")
    if case_id in results:
        continue

    result = Result()

    exp = full_atmo(**kwargs, backend="mitsuba")
    result.mitsuba = eradiate.run(exp, spp=SPP)["radiance"].squeeze()

    exp = full_atmo(**kwargs, backend="disort")
    backend = ed.EradiateDisortBackend(intensity_correction="buras_emde")
    result.disort = disort_reshape_pplane(backend.run(exp).sel(z=1e5))

    results[case_id] = result

# %%
ncases = len(CASES)
ncols = 2
nrows = ncases // ncols + min(ncases % ncols, 1)

fig, axs = plt.subplots(
    nrows, ncols, figsize=(4 * ncols, 3 * nrows), layout="constrained", squeeze=False
)

for i, case_id in enumerate(CASES.keys()):
    irow = i // ncols
    icol = i % ncols
    ax = axs.ravel()[i]
    result = results[case_id]

    ax.plot(result.mitsuba["vza"], result.mitsuba, label="Mitsuba" if i == 0 else None)
    ax.plot(
        result.disort["vza"],
        result.disort,
        label="CDISORT" if i == 0 else None,
        ls="--",
    )

    ax.set_title(case_id)
    ax.set_xlabel("θ [°]")
    if icol == 0:
        ax.set_ylabel("Radiance [W/m²/sr]")

else:
    while (i := i + 1) < axs.size:
        ax = axs.ravel()[i]
        ax.set_axis_off()

fig.legend(title="Backend", ncol=2, loc="outside upper center")

plt.show()
