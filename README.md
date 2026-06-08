# Eradiate DISORT backend

[![PyPI version](https://img.shields.io/pypi/v/eradiate-disort?color=blue)](https://pypi.org/project/eradiate-disort)
[![GitHub Workflow Status (branch)](https://img.shields.io/github/actions/workflow/status/eradiate/eradiate-disort/test.yml?branch=main)](https://github.com/eradiate/eradiate-disort/actions/workflows/test.yml)
[![Documentation Status](https://img.shields.io/readthedocs/eradiate-disort)](https://eradiate.readthedocs.io/projects/disort/latest)

A DISORT radiometric backend for the [Eradiate](https://github.com/eradiate/eradiate) radiative transfer model.

## Overview

This project lets Eradiate experiments be solved with DISORT instead of Monte
Carlo ray tracing, providing a fast radiometric backend for 1D plane-parallel
scenes. The solver is reached through the [nanodisort](https://github.com/eradiate/nanodisort)
bindings to CDISORT, a C port of the original DISORT software.

## Main features

- Computes radiances and fluxes for 1D plane-parallel atmospheres (no
  polarization).
- Drives an Eradiate experiment end-to-end, returning results as an
  `xarray.DataTree`.
- Supports Eradiate's `mono` and `ckd` spectral modes.
- Defines a `DisortMeasure` to record fluxes and intensities at chosen altitudes
  or optical thicknesses, optionally as a full radiance field.

## Documentation

See the [online documentation](https://eradiate-disort.readthedocs.io) for
installation details, the API reference and worked examples.

## Requirements and installation

- Linux or macOS, Python 3.10 to 3.13.
- A working Eradiate installation.

```bash
pip install eradiate[kernel,recommended] eradiate-disort
```

## Usage

Configure an Eradiate experiment with a `disort` measure, then run it through the
backend:

```python
import eradiate
import eradiate_disort as ed
from eradiate.experiments import AtmosphereExperiment

eradiate.set_mode("ckd")

exp = AtmosphereExperiment(
    surface={"type": "lambertian", "reflectance": 0.5},
    atmosphere=None,
    illumination={"type": "directional", "zenith": 30.0, "azimuth": 0.0},
    measures={
        "type": "disort",
        "construct": "hplane",
        "azimuth": 0.0,
        "zeniths": [-60.0, -30.0, 0.0, 30.0, 60.0],
        "srf": {"type": "delta", "wavelengths": [550.0]},
    },
)

backend = ed.DisortBackend()
result = backend.run(exp)
```

## About

Developed and maintained by Vincent Leroy and Claudia Emde.

This backend builds on CDISORT and DISORT; see the
[nanodisort README](https://github.com/eradiate/nanodisort) for the full credits
to the CDISORT and DISORT authors.

Consistent with the nanodisort package it links to, this project is licensed
under the terms of the GNU General Public License v3.0 or later
(GPL-3.0-or-later).
