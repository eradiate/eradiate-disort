# CLAUDE.md

This file provides guidance to coding agents when working with code in this repository.

## Overview

`eradiate-disort` provides a DISORT radiometric backend for the [Eradiate](https://github.com/eradiate/eradiate) radiative transfer model. It computes radiances and fluxes for 1D plane-parallel scenes using CDISORT (a C port of DISORT) as a fast alternative to Eradiate's Monte Carlo ray tracing backend. The DISORT solver is reached through the [`nanodisort`](https://github.com/eradiate/nanodisort) Python bindings. License: GPLv3 (inherited from nanodisort/CDISORT).

## Environment & dependencies

The project is managed with **pixi**. `eradiate` is pulled from PyPI (with the `kernel` and `recommended` extras, so the Mitsuba kernel ships as a prebuilt wheel — no local build). One dependency is still a git submodule under `ext/` and installed as editable:

- `ext/nanodisort` — CDISORT bindings (editable)

`[project.dependencies]` still declares a released version range for `nanodisort`, which the submodule (a release candidate, say) may not satisfy. `[tool.pixi.feature.dev.pypi-options].dependency-overrides` rewrites that requirement to the local path so the `dev` environment tracks the submodule whatever it is pinned to; the `[tool.pixi.feature.dev.pypi-dependencies]` entry is what makes the install editable, so both are needed.

After cloning, initialise the submodule (`git submodule update --init --recursive`). The pixi `dev` activation runs `scripts/set-macos-deployment-target.sh` (no-op off macOS; pins `MACOSX_DEPLOYMENT_TARGET` so locally built `nanodisort` wheel tags match uv's acceptance ceiling) and sets `ERADIATE_PATH=tests/data`.

Environments: `default` (`py310` + `test` + `docs` + `dev` — the working environment) plus the CI environments `ghapy310`–`ghapy313` (`pyXXX` + `kernel` + `test`, one per supported interpreter).

## Common commands

Run via `pixi run <task>`. Tasks are defined per feature in `pyproject.toml`: `test` lives under `[tool.pixi.feature.test.tasks]`; the rest (`bench`, `lint*`, `docs*`, `nb-*`, `bump*`) under `[tool.pixi.feature.dev.tasks]`. There is no top-level `[tool.pixi.tasks]` table.

- `pixi run test` — run the test suite (sets `MPLBACKEND=Agg`, `ERADIATE_TEST_MODE=test`). Benchmarks are excluded by default.
- `pixi run bench` — run benchmarks in `tests/benchmarks` (`ERADIATE_TEST_MODE=benchmark`).
- `pixi run lint` — `ruff check`. `pixi run lint-ext` adds extended rule sets (C4, SIM, PIE, PERF, NPY, RUF) on top of the configured ones. `pixi run lint-reuse` runs `reuse lint` (SPDX/license compliance).
- `pixi run docs` / `docs-serve` / `docs-clean` — build / live-serve / clean Sphinx docs.
- `pixi run nb-execute-all` — execute the example notebooks (`ERADIATE_TEST_MODE=tutorial`).

Run a single test: `pixi run test tests/test_backend.py::test_name` (the `test` task wraps `pytest`, so extra args pass through).

Regenerate regression reference data: append `--force-regen` (pytest-regressions). Reference `.npz` files live next to their test module (e.g. `tests/test_backend/`).

## Architecture

### Execution flow

`DisortBackend.run()` (`src/eradiate_disort/_backend.py`) drives three stages, mirroring Eradiate's experiment lifecycle:

1. **`validate(exp)`** — rejects unsupported configurations early. The backend only supports: `DirectionalIllumination`, `LambertianBSDF` surface, atmospheres in `{Molecular, ParticleLayer, Heterogeneous, Homogeneous}`, `DisortMeasure` measures, and at most one radiance-mode measure (DISORT has a single shared `umu`/`phi` grid). Both scene geometries are accepted; a `SphericalShellGeometry` sets `spher`/`radius`/`zd` and enables CDISORT's pseudo-spherical correction of the direct beam.
2. **`process(exp)`** — `_setup_global` (spectral-independent setup, classifies measures, sets illumination/control flags, sizes the phase grid) then a spectral loop calling `_setup_spectral` → `_solve` → `_collect_results` per spectral context. Raw per-spectral DISORT output dicts are accumulated in `self._results`.
3. **`postprocess(exp)`** — runs the pipeline (`_pipeline.py`) to convert raw results into an `xarray.DataTree` with one subtree per measure (keyed by measure ID).

All measures share one spectral loop; the loop's spectral grid comes from a single driving `measure` (default: the first).

### Key conventions and gotchas (DISORT ↔ Eradiate translation)

These are the subtle parts; the actual translation lives in `_backend.py` and `_phase.py`:

- **Ordering**: DISORT expects layers and moments **top-to-bottom**; Eradiate works bottom-to-top. Arrays are reversed at the boundary (`dtauc[::-1]`, `ssalb[::-1]`, `pmom[:, ::-1]`).
- **Azimuth**: DISORT `phi0` is the beam's *travel* direction; Eradiate's `illumination.azimuth` is the *source* direction — they differ by 180°.
- **Cumulative optical depth (`utau`)** is measured from TOA (0 at TOA, total τ at BOA). `DisortMeasure` accepts either `z_levels` (snapped to grid boundaries) or `utau` (mutually exclusive); see `_utau_from_spec` in `_measurements.py`.
- **Flux-only mode** (`onlyfl=True`, no `direction_layout`): CDISORT internally overwrites `numu` with `nstr`, so user `numu`/`nphi`/`umu`/`phi` must be re-assigned on every spectral iteration after the first.
- **Intensity correction**: `buras_emde` (default) needs actual phase function values and pads the μ-phase grid with sentinel points at both ends (`±(1+eps)`); the `+2` to `nphase` in `_setup_global` accounts for these. `nakajima_tanaka` uses only Legendre moments. See the `project_buras_emde` memory.
- **Phase moments**: Eradiate's particle data stores `(2l+1)·f_l`; DISORT wants `f_l`, so `_phase.py` divides by `(2l+1)` and truncates/zero-pads to `nmom+1`.
- **Homogeneous atmospheres** return scalar optical properties — broadcast them to `nlyr` before assigning.
- **Pseudo-spherical geometry**: `spher` and `radius` are scalars set in `_setup_global`; `zd` is an array and can only be assigned after `allocate()`, so it lives in `_setup_spectral`. `zd` holds level heights above the *ground surface*, top-to-bottom, with `zd[nlyr]` exactly `0` (nanodisort checks this). `radius` is `planet_radius + ground_altitude` (that is the radius of the sphere `zd = 0` sits on), in the same length unit as `zd` (km).
- **Allocation**: `ds.allocate()` is called exactly once (`first_call=True`), after `ntau` is known and before any array assignment. Do not assign DISORT arrays before allocation.

### Components

- `_backend.py` — `DisortBackend` (the entry point).
- `_measurements.py` — `DisortMeasure`, a custom Eradiate `Measure` registered in `measure_factory` as `"disort"`. Records fluxes/intensities at altitudes/optical depths, optionally a full radiance field. Convenience constructors: `hplane`, `aring`, `grid`.
- `_phase.py` — `get_pmom` (Legendre moments) and `get_phase` (μ-grid + phase values) for atmospheres, dispatching on phase-function type.
- `_pipeline.py` — Eradiate `Pipeline` DAG (`build_disort_pipeline`): `stacked_data` → `aggregated_data` (CKD quadrature) → `datatree`.
- `io.py` — `normalize_metadata` for CF-style coordinate/variable attributes.
- `util.py` — post-processing helpers for run output (e.g. `disort_reshape_pplane`, which rebuilds a signed-zenith principal-plane radiance `DataArray` from a result `DataTree`).
- `testing/` — shared helpers (importable as `eradiate_disort.testing`): `TestMode` (mode selector via `ERADIATE_TEST_MODE`), `cases` (canonical experiment builders), `xarray_regression` fixture, plotting helpers.

### Spectral modes

The backend supports Eradiate's `mono` and `ckd` modes (set via `eradiate.set_mode(...)`). Mode-dependent branches appear in `_get_spectral_indices` (mono: wavelength floats; ckd: `(w, g)` tuples with quadrature) and in pipeline aggregation.

## Testing & examples

- Tests live in `tests/`; `tests/conftest.py` pulls in fixtures, the `er_plt` plotting fixture, and `xarray_regression`.
- **Examples double as regression tests.** `tests/examples/example_*.py` are jupytext percent-format notebooks (paired to `docs/examples/*.ipynb`). `tests/examples/test_examples.py` imports each example's `result` and asserts (some are still smoke tests — see `TODO.md`). These run *last* (`pytest.mark.order(-1)`).
- `TestMode.spp(...)` / `TestMode.plt()` let one notebook serve tutorial (interactive, low SPP), test (full SPP, no plots), and benchmark modes. Don't hardcode sample counts in shareable notebooks.
- Regression checks use the `xarray_regression` fixture (wraps `ndarrays_regression`); flatten coords are prefixed `coord_`. Pass tolerances via `default_tolerance`/`tolerances`.
- Benchmarks (`tests/benchmarks/bench_*.py`) use a separate `pytest.ini` and pytest-benchmark; functions/classes use the `bench_`/`Bench` prefix.

## Docs

Sphinx + shibuya theme + MyST-NB, deployed on Read The Docs. Example notebook outputs are **committed** (nbstripout excludes `docs/examples/`) so MyST-NB renders them without executing. `docs/requirements.txt` is regenerated with `pixi run docs-lock`.

## Conventions

- Ruff for lint+format (rules `E`, `F`, `I`, `B`, `UP`; example notebooks excluded). Numpydoc docstrings. Type hints in signatures. Private modules are underscore-prefixed; public API is re-exported from `__init__.py` (`DisortBackend`, `DisortMeasure`).
- All source files carry SPDX headers (`GPL-3.0-or-later`, copyright Rayference); REUSE-compliant.
- pre-commit runs the standard pre-commit-hooks set, ruff (check+format), zizmor (GitHub Actions), taplo (TOML), nbstripout, and bibtex-tidy.
