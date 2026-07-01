# Eradiate DISORT — Changelog

---

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
