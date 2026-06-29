# Eradiate DISORT — Changelog

---

## v0.2.0 (upcoming release)

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
