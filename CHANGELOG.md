# Eradiate DISORT — Changelog

---

## v1.2.0 (5th November 2026)

This release bumps the required Eradiate version to 1.3.0:

- The eradiate dependency is bumped to 1.3.0.
- Python 3.14 is officially supported.
- The code (including examples) is adapted to the breaking API changes in
  Eradiate v1.3.0:
  - Rename `ParticleLayer` to `ParticleOTComponent`.
  - Use `geometry.grid` (`SpatialGrid`) instead of `geometry.zgrid` (`ZGrid`);
    build grids with `CartesianGrid.from_levels`.
  - Flatten `(x, y, z)` radiative property arrays to 1D.
  - Reject atmosphere grids that are not a single column.
  - Use `particle_components` instead of the deprecated `particle_layers`
    parameter.
- In many examples, the Monte Carlo reference sample count is increased tenfold
  to adapt to the new sample allocation policy implemented by Eradiate v1.3.0.

## v1.1.0 (7th September 2026)

### Added

- Support for spherical-shell geometries: an experiment configured with a
  `SphericalShellGeometry` now enables CDISORT's pseudo-spherical correction,
  which applies the Chapman airmass to the direct beam. Plane-parallel
  experiments are unaffected.

## Improved

- nanodisort is updated to v0.3.0, which brings pseudo-spherical correction
  support.

## v1.0.0 (1st July 2026)

This is a milestone release that marks the transition of this package to a
stable status. The main API entry points should not change significantly.

### Fixed

- Dither the solar zenith angle slightly when it coincides with an angular
  quadrature node, which CDISORT rejects (similar to libRadtran)
  ([#4](https://github.com/eradiate/eradiate-disort/pull/4)).
- Convert the solar azimuth angle in the output `DataTree` to the Eradiate
  convention (180° flip) instead of emitting the raw backend value
  ([#4](https://github.com/eradiate/eradiate-disort/pull/4)).

## v0.1.0 (10th June 2026)

*First release.*

This release bootstraps the DISORT backend. It provides the features needed to
compute radiances and fluxes using `AtmosphereExperiment`.
