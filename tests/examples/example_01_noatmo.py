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
import eradiate
import matplotlib.pyplot as plt

import eradiate_disort as ed
from eradiate_disort.testing import TestMode, cases
from eradiate_disort.testing.util import Result, disort_reshape_pplane

eradiate.set_mode("ckd")

SPP = 1_000

# %% tags=["remove-cell"]
# Dev-specific setup, hidden from notebook output

plt = TestMode.plt()
_base_spp = TestMode.spp(tutorial=1_000, test=10_000)
SPP = _base_spp // 16 if eradiate.get_mode().is_ckd else _base_spp

# %%
result = Result()

exp = cases.no_atmo(30.0, backend="mitsuba")
result.mitsuba = eradiate.run(exp, spp=SPP)["radiance"].squeeze()

exp = cases.no_atmo(30.0, backend="disort")
backend = ed.DisortBackend()
result.disort = disort_reshape_pplane(backend.run(exp).sel(z=1.0))

# %%
fig, ax = plt.subplots(1, 1, figsize=(4, 3), layout="constrained")

ax.plot(result.mitsuba["vza"], result.mitsuba, label="Mitsuba")
ax.plot(result.disort["vza"], result.disort, label="CDISORT", ls="--")
ax.set_xlabel("θ [°]")
ax.set_ylabel("Radiance [W/m²/sr]")
ax.set_ylim([-0.05, 0.65])
ax.legend()

plt.show()
